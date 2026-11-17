from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from hazy import Frame, Point, Points, Vector, Vectors

coordinate = st.floats(min_value=-100, max_value=100, allow_nan=False)
batch_coords = arrays(
    float,
    st.tuples(st.integers(1, 8), st.just(3)),
    elements=coordinate,
)


@pytest.fixture
def camera(root_frame: Frame) -> Frame:
    return (
        root_frame.make_child("camera")
        .translate(x=1, y=-2, z=3)
        .rotate_euler(x=20, y=-30, z=45, degrees=True)
        .scale(2.0)
    )


@pytest.fixture
def cloud(camera: Frame) -> Points:
    return camera.points(np.arange(12.0).reshape(4, 3))


@pytest.mark.unit
class TestConstruction:
    def test_copies_input_by_default(self, camera: Frame):
        coords = np.zeros((2, 3))
        cloud = camera.points(coords)
        coords[0, 0] = 5.0
        assert cloud[0].x == 0.0

    def test_copy_false_shares_memory(self, camera: Frame):
        coords = np.zeros((2, 3))
        cloud = camera.points(coords, copy=False)
        coords[0, 0] = 5.0
        assert cloud[0].x == 5.0

    @pytest.mark.parametrize("shape", [(3,), (4, 2), (2, 4)])
    def test_rejects_invalid_shapes_with_hint(self, camera: Frame, shape):
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
            camera.points(np.zeros(shape))

    def test_from_primitives_round_trips_to_list(self, camera: Frame):
        points = [camera.point(1, 2, 3), camera.point(4, 5, 6)]
        cloud = Points.from_primitives(points)
        assert cloud.shape == (2,)
        assert all(a == b for a, b in zip(cloud.to_list(), points, strict=True))

    def test_from_primitives_rejects_mixed_kinds(self, camera: Frame):
        with pytest.raises(TypeError, match="expects only Point"):
            Points.from_primitives([camera.point(1, 2, 3), camera.vector(1, 2, 3)])

    def test_from_primitives_rejects_mixed_frames(self, camera: Frame):
        other = camera.make_child("other")
        with pytest.raises(RuntimeError, match="same coordinate system"):
            Vectors.from_primitives([camera.vector(1, 0, 0), other.vector(1, 0, 0)])

    def test_from_primitives_rejects_empty_sequence(self):
        with pytest.raises(ValueError, match="empty"):
            Points.from_primitives([])


@pytest.mark.unit
class TestIndexing:
    def test_single_element_is_independent_copy(self, cloud: Points):
        element = cloud[1]
        element.round_().clip_(a_max=0)
        assert isinstance(element, Point)
        assert element.frame is cloud.frame
        assert_allclose(cloud[1], [3, 4, 5])

    def test_vectors_element_is_vector(self, camera: Frame):
        assert isinstance(camera.vectors(np.ones((2, 3)))[0], Vector)

    def test_slice_is_view_like_numpy(self, cloud: Points):
        cloud[1:3].floor_()
        cloud[1:3].clip_(a_max=5)
        assert_allclose(np.asarray(cloud)[1:3], [[3, 4, 5], [5, 5, 5]])

    def test_mask_selection_is_copy_like_numpy(self, cloud: Points):
        selected = cloud[cloud.z > 4]
        selected.clip_(a_max=0)
        assert selected.shape == (3,)
        assert_allclose(cloud[3], [9, 10, 11])

    def test_index_never_reaches_coordinate_axis(self, cloud: Points):
        with pytest.raises(IndexError):
            cloud[0, 1]

    def test_multidimensional_batches_index_leading_axes(self, camera: Frame):
        grid = camera.points(np.zeros((4, 5, 3)))
        assert grid.shape == (4, 5)
        assert grid[0].shape == (5,)
        assert isinstance(grid[0, 1], Point)
        assert grid[..., 0].shape == (4,)
        assert len(grid.to_list()) == 20

    def test_iteration_yields_elements(self, cloud: Points):
        assert [p.x for p in cloud] == [0, 3, 6, 9]

    def test_setitem_accepts_same_kind_and_frame(self, cloud: Points, camera: Frame):
        cloud[0] = camera.point(-1, -1, -1)
        cloud[1:3] = camera.points([[7, 7, 7]])
        assert_allclose(np.asarray(cloud)[:3], [[-1] * 3, [7] * 3, [7] * 3])

    def test_setitem_rejects_wrong_kind_and_raw_arrays(
        self, cloud: Points, camera: Frame
    ):
        with pytest.raises(TypeError, match="np.asarray"):
            cloud[0] = [1, 2, 3]
        with pytest.raises(TypeError, match="Can only assign Point or Points"):
            cloud[0] = camera.vector(1, 2, 3)

    def test_setitem_rejects_other_frame(self, cloud: Points, root_frame: Frame):
        with pytest.raises(RuntimeError, match="same coordinate system"):
            cloud[0] = root_frame.point(1, 2, 3)

    def test_coordinate_properties_are_readonly(self, cloud: Points):
        assert_allclose(cloud.y, [1, 4, 7, 10])
        with pytest.raises(ValueError, match="read-only"):
            cloud.x[0] = 1.0


@pytest.mark.integration
class TestTransformations:
    @settings(max_examples=50, deadline=None)
    @given(coords=batch_coords)
    def test_batched_transform_matches_single_primitives(self, coords):
        world = Frame.make_root("world")
        arm = world.make_child("arm").translate(x=1).rotate_euler(z=30, degrees=True)
        tool = arm.make_child("tool").scale(1, 2, 3).translate(y=-2)
        sibling = world.make_child("sibling").rotate_euler(x=90, degrees=True)

        for batch_type, single_type in ((Points, Point), (Vectors, Vector)):
            batch = batch_type.from_array(coords, frame=tool).to_frame(sibling)
            expected = [
                single_type.from_array(row, frame=tool).to_frame(sibling)
                for row in coords
            ]
            assert batch.frame is sibling
            assert_allclose(batch, np.stack(expected), atol=1e-9)

    def test_to_global_and_back_is_identity(self, cloud: Points, camera: Frame):
        assert cloud.to_global().to_frame(camera).isclose(cloud).all()

    def test_vectors_ignore_translation(self, root_frame: Frame):
        frame = root_frame.make_child("moved").translate(x=10)
        assert_allclose(frame.vectors([[1, 0, 0]]).to_global(), [[1, 0, 0]])
        assert_allclose(frame.points([[1, 0, 0]]).to_global(), [[11, 0, 0]])

    def test_isclose_compares_in_global_frame(self, cloud: Points):
        assert cloud.isclose(cloud.to_global()).all()
        assert not cloud.isclose(cloud.to_global()[0]).all()

    def test_isclose_rejects_other_kind(self, cloud: Points):
        with pytest.raises(TypeError, match="Cannot compare"):
            cloud.isclose(cloud - cloud)

    def test_equality_operator_points_to_isclose(self, cloud: Points):
        with pytest.raises(TypeError, match=r"isclose"):
            cloud == cloud.copy()


@pytest.mark.unit
class TestArithmetic:
    @pytest.mark.parametrize(
        ("expression", "expected_type"),
        [
            ("points - points", Vectors),
            ("points - point", Vectors),
            ("point - points", Vectors),
            ("points + vectors", Points),
            ("points + vector", Points),
            ("point + vectors", Points),
            ("vectors + points", Points),
            ("vector + points", Points),
            ("points - vectors", Points),
            ("point - vectors", Points),
            ("vectors + vectors", Vectors),
            ("vectors - vector", Vectors),
            ("vector - vectors", Vectors),
            ("-vectors", Vectors),
            ("vectors * 2", Vectors),
            ("2 * vectors", Vectors),
            ("np.float64(2) * vectors", Vectors),
            ("vectors / 2", Vectors),
        ],
    )
    def test_result_types(self, camera: Frame, expression: str, expected_type):
        namespace = {
            "np": np,
            "points": camera.points(np.ones((3, 3))),
            "vectors": camera.vectors(np.ones((3, 3))),
            "point": camera.point(1, 2, 3),
            "vector": camera.vector(1, 2, 3),
        }
        result = eval(expression, namespace)
        assert type(result) is expected_type
        assert result.frame is camera

    @pytest.mark.parametrize(
        ("expression", "error", "message"),
        [
            ("points + points", TypeError, "geometrically undefined"),
            ("point + points", TypeError, "geometrically undefined"),
            ("vectors - points", TypeError, "geometrically undefined"),
            ("vector - points", TypeError, "geometrically undefined"),
            ("points * 2", TypeError, "relative to an origin"),
            ("points / 2", TypeError, "relative to an origin"),
            ("2 / vectors", TypeError, "undefined"),
            ("vectors * vectors", TypeError, "dot"),
            ("vectors / 0", ZeroDivisionError, "zero"),
            ("vectors * 1j", TypeError, "real numbers"),
            ("vectors * np.ones((3, 3))", ValueError, "per-component"),
            ("points + np.ones((3, 3))", TypeError, "frame.vectors"),
            ("np.ones((3, 3)) + points", TypeError, "frame.vectors"),
        ],
    )
    def test_invalid_operations_explain_alternatives(
        self, camera: Frame, expression: str, error, message: str
    ):
        namespace = {
            "np": np,
            "points": camera.points(np.ones((3, 3))),
            "vectors": camera.vectors(np.ones((3, 3))),
            "point": camera.point(1, 2, 3),
            "vector": camera.vector(1, 2, 3),
        }
        with pytest.raises(error, match=message):
            eval(expression, namespace)

    def test_mixed_frames_raise(self, cloud: Points, root_frame: Frame):
        with pytest.raises(RuntimeError, match="same coordinate system"):
            cloud - root_frame.points(np.zeros((4, 3)))

    def test_broadcasting_gives_pairwise_differences(self, camera: Frame):
        a = camera.points(np.zeros((4, 1, 3)))
        b = camera.points(np.ones((5, 3)))
        assert (a - b).shape == (4, 5)

    def test_per_element_scaling(self, camera: Frame):
        vectors = camera.vectors([[1, 0, 0], [0, 2, 0]])
        assert_allclose((vectors * [3, 0.5]).magnitude, [3, 1])

    def test_scaling_relative_to_origin(self, cloud: Points, camera: Frame):
        origin = camera.origin
        scaled = (cloud - origin) * 2 + origin
        assert_allclose(scaled, 2 * np.asarray(cloud))


@pytest.mark.unit
class TestVectorsOperations:
    def test_magnitude_and_normalize(self, camera: Frame):
        vectors = camera.vectors([[3, 4, 0], [0, 0, 2]])
        assert_allclose(vectors.magnitude, [5, 2])
        assert vectors.normalize() is vectors
        assert_allclose(vectors.magnitude, [1, 1])

    def test_normalize_rejects_zero_vectors_with_filter_hint(self, camera: Frame):
        vectors = camera.vectors([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(RuntimeError, match="is_zero"):
            vectors.normalize()
        assert_allclose(vectors[~vectors.is_zero].normalize(), [[1, 0, 0]])

    def test_dot_and_cross_broadcast_with_single_vector(self, camera: Frame):
        vectors = camera.vectors(np.eye(3))
        assert_allclose(vectors.dot(camera.x_axis), [1, 0, 0])
        assert_allclose(vectors.cross(camera.z_axis), [[0, -1, 0], [1, 0, 0], [0] * 3])

    def test_dot_rejects_points(self, camera: Frame):
        with pytest.raises(TypeError, match="requires Vector or Vectors"):
            camera.vectors(np.eye(3)).dot(camera.points(np.eye(3)))


@pytest.mark.numpy
class TestNumpyIntegration:
    def test_asarray_is_view_and_array_is_copy(self, cloud: Points):
        np.asarray(cloud)[0, 0] = 42.0
        np.array(cloud)[0, 0] = -1.0
        assert cloud[0].x == 42.0

    def test_asarray_rejects_integer_dtype(self, cloud: Points):
        with pytest.raises(TypeError, match="floating-point"):
            np.asarray(cloud, dtype=int)

    def test_rounding_preserves_type_and_frame(self, cloud: Points):
        shifted = cloud - cloud.frame.vector(0.4, 0.6, 0.5)
        for result in (
            np.round(shifted),
            np.floor(shifted),
            np.ceil(shifted),
            np.trunc(shifted),
            np.rint(shifted),
            np.fix(shifted),
            np.clip(shifted, 0, 5),
        ):
            assert type(result) is Points
            assert result.frame is cloud.frame
        assert_allclose(np.clip(cloud, a_max=4), np.clip(np.asarray(cloud), None, 4))

    def test_type_queries_return_coordinate_masks(self, camera: Frame):
        cloud = camera.points([[np.nan, 0, np.inf]])
        assert_allclose(np.isnan(cloud), [[True, False, False]])
        assert_allclose(np.isfinite(cloud), [[False, True, False]])

    def test_vector_magnitudes_via_numpy(self, camera: Frame):
        vectors = camera.vectors([[3, 4, 0]])
        assert_allclose(np.linalg.norm(vectors), [5])
        assert_allclose(np.absolute(vectors), [5])
        assert_allclose(np.vecdot(vectors, vectors), [25])
        assert type(np.cross(vectors, camera.z_axis)) is Vectors

    def test_ambiguous_numpy_functions_raise_with_hints(self, cloud: Points):
        vectors = cloud - cloud
        with pytest.raises(TypeError, match="vectors.dot"):
            np.dot(vectors, vectors)
        with pytest.raises(TypeError, match="ord, axis"):
            np.linalg.norm(vectors, axis=0)
        with pytest.raises(TypeError, match="only defined for Vectors"):
            np.linalg.norm(cloud)
        with pytest.raises(TypeError, match="np.asarray"):
            np.sin(cloud)
        with pytest.raises(TypeError, match="np.asarray"):
            np.sum(cloud)
        with pytest.raises(TypeError, match="undefined"):
            np.absolute(cloud)

    def test_concatenate_and_stack_use_element_axes(self, cloud: Points):
        assert np.concatenate([cloud, cloud]).shape == (8,)
        assert np.stack([cloud, cloud]).shape == (2, 4)
        assert np.stack([cloud, cloud], axis=-1).shape == (4, 2)
        grid = np.stack([cloud, cloud])
        assert np.concatenate([grid, grid], axis=-1).shape == (2, 8)
        assert np.concatenate([grid, grid], axis=None).shape == (16,)

    def test_concatenate_rejects_mixed_kinds_and_frames(
        self, cloud: Points, root_frame: Frame
    ):
        with pytest.raises(TypeError, match="single kind"):
            np.concatenate([cloud, cloud - cloud])
        with pytest.raises(RuntimeError, match="same coordinate system"):
            np.concatenate([cloud, cloud.to_global()])

    def test_copy_is_independent(self, cloud: Points):
        duplicate = np.copy(cloud)
        duplicate.round_(decimals=-2)
        assert type(duplicate) is Points
        assert cloud[3].x == 9
