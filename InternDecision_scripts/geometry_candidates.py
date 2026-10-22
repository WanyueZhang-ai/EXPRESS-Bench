"""Geometry-only exploration candidates and validated executor mapping."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


class CandidateMappingError(KeyError):
    """A model-selected ID is absent from the current executable snapshot."""


_ALLOWED_FIELDS = {
    "point", "distance_m", "bearing_rad", "expected_new_fraction",
    "novelty", "view_bearing_rad", "frontier_neighbors",
}


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _point(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        raise ValueError("candidate point must have three coordinates")
    return tuple(_number(item, "candidate point") for item in value)


@dataclass(frozen=True)
class ExecutorTarget:
    mode: str
    source_index: int
    point: tuple[float, float, float]


class GeometryCandidateGenerator:
    """Expose safe candidate fields while retaining the exact executable point."""

    def __init__(self, *, max_candidates: int = 8) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.max_candidates = int(max_candidates)
        self._targets: dict[str, ExecutorTarget] = {}

    def build(
        self,
        *,
        fbe: Iterable[Mapping[str, Any]],
        goe: Iterable[Mapping[str, Any]],
        path_checker: Callable[[tuple[float, float, float]], bool],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        targets: dict[str, ExecutorTarget] = {}
        for mode, sources in (("fbe", fbe), ("goe", goe)):
            for source_index, source in enumerate(sources):
                if not isinstance(source, Mapping):
                    raise ValueError("candidate must be a mapping")
                forbidden = set(source) - _ALLOWED_FIELDS
                if forbidden:
                    raise ValueError(f"unsupported candidate fields: {sorted(forbidden)}")
                point = _point(source.get("point"))
                if not path_checker(point):
                    continue
                identifier = f"{mode}_{sum(row['mode'] == mode for row in rows)}"
                row: dict[str, Any] = {"id": identifier, "mode": mode}
                for key in sorted(_ALLOWED_FIELDS - {"point"}):
                    if key in source:
                        row[key] = _number(source[key], key)
                rows.append(row)
                targets[identifier] = ExecutorTarget(mode, source_index, point)
                if len(rows) >= self.max_candidates:
                    self._targets = targets
                    return rows
        self._targets = targets
        return rows

    def executor_target(self, candidate_id: str) -> tuple[str, int]:
        target = self._targets.get(candidate_id)
        if target is None:
            raise CandidateMappingError(candidate_id)
        return target.mode, target.source_index

    def target_point(self, candidate_id: str) -> tuple[float, float, float]:
        target = self._targets.get(candidate_id)
        if target is None:
            raise CandidateMappingError(candidate_id)
        return target.point


def geometry_frontiers(
    free: np.ndarray,
    observed: np.ndarray,
    *,
    current_xy: Sequence[float],
    origin_xy: Sequence[float],
    voxel_size: float,
    floor_height: float,
    max_candidates: int = 8,
    min_spacing_m: float = 1.0,
) -> list[dict[str, Any]]:
    """Return reachable-free cells bordering unknown cells, with no semantics."""
    free = np.asarray(free, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if free.ndim != 2 or free.shape != observed.shape:
        raise ValueError("free and observed must be matching 2D maps")
    voxel_size = _number(voxel_size, "voxel_size")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    unknown = ~observed
    padded = np.pad(unknown.astype(np.int8), 1)
    neighbor_count = sum(
        padded[1 + dx:1 + dx + free.shape[0], 1 + dy:1 + dy + free.shape[1]]
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    cells = np.argwhere(free & observed & (neighbor_count > 0))
    current = np.asarray(current_xy, dtype=float)
    origin = np.asarray(origin_xy, dtype=float)
    if current.shape != (2,) or origin.shape != (2,):
        raise ValueError("current_xy and origin_xy must have two coordinates")
    ranked = []
    for cell in cells:
        xy = origin + cell * voxel_size
        distance = float(np.linalg.norm(xy - current))
        ranked.append((-int(neighbor_count[tuple(cell)]), distance, tuple(cell), xy))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    selected: list[dict[str, Any]] = []
    for negative_neighbors, distance, _, xy in ranked:
        if distance < voxel_size:
            continue
        if any(np.linalg.norm(xy - np.asarray(row["point"][:2])) < min_spacing_m for row in selected):
            continue
        selected.append({
            "point": [float(xy[0]), float(xy[1]), float(floor_height)],
            "distance_m": distance,
            "bearing_rad": float(math.atan2(xy[1] - current[1], xy[0] - current[0])),
            "frontier_neighbors": -negative_neighbors,
            "expected_new_fraction": min(1.0, -negative_neighbors / 4.0),
            "novelty": 1.0,
        })
        if len(selected) >= max_candidates:
            break
    return selected


def observed_viewpoints(
    free: np.ndarray,
    observed: np.ndarray,
    *,
    current_xy: Sequence[float],
    origin_xy: Sequence[float],
    voxel_size: float,
    floor_height: float,
    max_candidates: int = 4,
    min_distance_m: float = 1.0,
) -> list[dict[str, Any]]:
    """Suggest fair re-observation goals from already seen, free geometry."""
    free = np.asarray(free, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if free.ndim != 2 or free.shape != observed.shape:
        raise ValueError("free and observed must be matching 2D maps")
    current = np.asarray(current_xy, dtype=float)
    origin = np.asarray(origin_xy, dtype=float)
    candidates = []
    for cell in np.argwhere(free & observed):
        xy = origin + cell * voxel_size
        delta = xy - current
        distance = float(np.linalg.norm(delta))
        if distance < min_distance_m:
            continue
        angle = float(math.atan2(delta[1], delta[0]))
        candidates.append((distance, tuple(cell), xy, angle))
    candidates.sort(key=lambda item: (item[0], item[1]))
    # One deterministic viewpoint per angular sector. The backbone decides
    # whether any of these question-conditioned re-observation options has value.
    sectors: dict[int, tuple] = {}
    for item in candidates:
        sector = int(((item[3] + math.pi) / (2 * math.pi)) * 8) % 8
        sectors.setdefault(sector, item)
    result = []
    for sector in sorted(sectors):
        distance, _, xy, angle = sectors[sector]
        result.append({
            "point": [float(xy[0]), float(xy[1]), float(floor_height)],
            "distance_m": distance,
            "bearing_rad": angle,
            "view_bearing_rad": angle,
            "expected_new_fraction": 0.0,
            "novelty": 0.5,
        })
        if len(result) >= max_candidates:
            break
    return result
