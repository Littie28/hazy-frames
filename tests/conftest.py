from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "numpy: NumPy integration tests for geometric primitives"
    )
    config.addinivalue_line("markers", "unit: Unit tests for core functionality")
    config.addinivalue_line(
        "markers", "integration: test the interplay between different components"
    )
