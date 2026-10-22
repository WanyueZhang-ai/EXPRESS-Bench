"""Bounded movement along a geometry-only Habitat PathFinder route."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class GeometryNavigationError(ValueError):
    """No safe geometry route can be executed."""


@dataclass(frozen=True)
class NavigationStep:
    position: tuple[float, float, float]
    distance_m: float
    reached_target: bool


def _point(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise GeometryNavigationError("route points must be 3D")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise GeometryNavigationError("route points must be finite")
    return result


def bounded_waypoint(route: Sequence[Sequence[float]], *, max_step_m: float) -> NavigationStep:
    """Interpolate a single waypoint at most ``max_step_m`` along a route."""
    if not route:
        raise GeometryNavigationError("route is empty")
    limit = float(max_step_m)
    if not math.isfinite(limit) or limit <= 0:
        raise GeometryNavigationError("max_step_m must be positive and finite")
    points = [_point(point) for point in route]
    travelled = 0.0
    previous = points[0]
    for next_point in points[1:]:
        length = math.dist(previous, next_point)
        if length <= 1e-9:
            previous = next_point
            continue
        if travelled + length >= limit:
            fraction = (limit - travelled) / length
            waypoint = tuple(a + fraction * (b - a) for a, b in zip(previous, next_point))
            reached = fraction >= 1.0 - 1e-9 and next_point == points[-1]
            return NavigationStep(waypoint, limit, reached)
        travelled += length
        previous = next_point
    return NavigationStep(points[-1], travelled, True)


def yaw_toward(current_xy: Sequence[float], target_xy: Sequence[float], *, current_yaw: float) -> float:
    """Face a geometry target; rotate 90 degrees when the target is current pose."""
    if len(current_xy) != 2 or len(target_xy) != 2:
        raise GeometryNavigationError("heading points must be 2D")
    dx = float(target_xy[0]) - float(current_xy[0])
    dy = float(target_xy[1]) - float(current_xy[1])
    if not all(math.isfinite(value) for value in (dx, dy, current_yaw)):
        raise GeometryNavigationError("heading values must be finite")
    if math.hypot(dx, dy) <= 1e-6:
        return float(current_yaw) + math.pi / 2
    return math.atan2(dy, dx) - math.pi / 2
