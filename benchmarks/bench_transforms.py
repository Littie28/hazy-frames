"""Micro benchmarks for transformation hot paths.

Run with: uv run python benchmarks/bench_transforms.py
"""

from __future__ import annotations

import timeit

import numpy as np

import hazy
from hazy import Frame


def build_hierarchy() -> tuple[Frame, Frame]:
    world = Frame.make_root(name="world")
    arm = world.make_child("arm").translate(x=1).rotate_euler(z=30, degrees=True)
    tool = arm.make_child("tool").translate(y=2).scale(2.0)
    return world, tool


def report(label: str, statement, number: int, per: int = 1) -> None:
    seconds = min(timeit.repeat(statement, number=number, repeat=5)) / number / per
    print(f"{label:<45} {seconds * 1e9:12.1f} ns")


def main() -> None:
    world, tool = build_hierarchy()
    point = tool.point(1, 2, 3)
    vector = tool.vector(1, 2, 3)
    coords = np.random.default_rng(0).random((1_000_000, 3))
    matrix = tool.transform_to(world)

    report("Frame.transform_to (cached)", lambda: tool.transform_to(world), 20_000)
    report("Frame.point construction", lambda: tool.point(1, 2, 3), 20_000)
    report("Point.to_frame", lambda: point.to_frame(world), 20_000)
    report("Vector.to_frame", lambda: vector.to_frame(world), 20_000)
    report("Point - Point", lambda: point - point, 20_000)
    report(
        "raw numpy per point (reference)",
        lambda: coords @ matrix[:3, :3].T + matrix[:3, 3],
        5,
        per=len(coords),
    )
    if hasattr(hazy, "Points"):
        cloud = tool.points(coords)
        report(
            "Points.to_frame per point", lambda: cloud.to_frame(world), 5, len(coords)
        )
        report("Points - Points per point", lambda: cloud - cloud, 5, len(coords))


if __name__ == "__main__":
    main()
