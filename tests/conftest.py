from __future__ import annotations

import numpy as np
import pytest
from hypothesis import strategies as st

from hazy import Frame


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "numpy: NumPy integration tests for geometric primitives"
    )
    config.addinivalue_line("markers", "unit: Unit tests for core functionality")
    config.addinivalue_line(
        "markers", "integration: test the interplay between different components"
    )


@pytest.fixture(scope="function")
def frame(root_frame: Frame) -> Frame:
    child = root_frame.make_child("child")
    return child


@pytest.fixture(scope="function")
def root_frame() -> Frame:
    root = Frame.make_root("root")
    return root


@st.composite
def translations(draw):
    """Strategy for generating translation vectors."""
    x = draw(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
    y = draw(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
    z = draw(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
    return np.array([x, y, z], dtype=float)


@st.composite
def rotations(draw):
    """Strategy for generating valid rotation matrices via Euler angles."""
    x = draw(
        st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    y = draw(
        st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    z = draw(
        st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
    )
    seq = draw(st.sampled_from(["xyz", "xzy", "yzx", "yxz", "zxy", "zyx"]))
    return {"x": x, "y": y, "z": z, "seq": seq, "degrees": True}


@st.composite
def scalings(draw):
    """Strategy for generating scaling vectors (positive values only)."""
    uniform = draw(st.booleans())
    if uniform:
        scale = draw(
            st.floats(
                min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
            )
        )
        return scale
    else:
        x = draw(
            st.floats(
                min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
            )
        )
        y = draw(
            st.floats(
                min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
            )
        )
        z = draw(
            st.floats(
                min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
            )
        )
        return np.array([x, y, z], dtype=float)


@st.composite
def translation_lists(draw, min_size=0, max_size=10):
    """Strategy for generating lists of translations."""
    return draw(st.lists(translations(), min_size=min_size, max_size=max_size))


@st.composite
def rotation_lists(draw, min_size=0, max_size=10):
    """Strategy for generating lists of rotations."""
    return draw(st.lists(rotations(), min_size=min_size, max_size=max_size))


@st.composite
def scaling_lists(draw, min_size=0, max_size=10):
    """Strategy for generating lists of scalings."""
    return draw(st.lists(scalings(), min_size=min_size, max_size=max_size))


@st.composite
def vectors_3d(draw):
    x = draw(st.floats(min_value=-100, max_value=100))
    y = draw(st.floats(min_value=-100, max_value=100))
    z = draw(st.floats(min_value=-100, max_value=100))
    return np.array([x, y, z])


@st.composite
def vector_3d_lists(draw, min_size=0, max_size=10):
    return draw(st.lists(vectors_3d(), min_size=min_size, max_size=max_size))
