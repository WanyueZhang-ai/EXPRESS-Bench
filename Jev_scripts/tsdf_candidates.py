"""Adapters that turn Fine-EQA planner state into stable Jev candidate records."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


def summarize_goe(
    exp_list: Iterable[Mapping[str, Any]],
    *,
    region_names: Optional[Mapping[int, str]] = None,
    max_candidates: int = 8,
) -> List[Dict[str, Any]]:
    """Summarize ``TSDFPlanner.exp_list`` without exposing the map arrays."""
    region_names = region_names or {}
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(exp_list or []):
        region_id = item.get("region")
        points = item.get("point") or []
        region = region_names.get(region_id, region_id)
        result.append(
            {
                "id": "goe_{}".format(index),
                "region": region,
                "region_id": region_id,
                "count": item.get("count"),
                "direction": item.get("dir"),
                "point_count": len(points) if isinstance(points, (list, tuple)) else None,
                "point_available": bool(points),
            }
        )
    return result[:max_candidates]


def summarize_fbe(
    candidates: Iterable[Mapping[str, Any]], *, max_candidates: int = 8
) -> List[Dict[str, Any]]:
    """Normalize frontier candidates while preserving their planner order."""
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(candidates or []):
        if not isinstance(item, Mapping):
            raise ValueError("FBE candidate must be a mapping")
        normalized = {str(key): value for key, value in item.items()}
        normalized["id"] = "fbe_{}".format(index)
        result.append(normalized)
    return result[:max_candidates]


def summarize_fbe_from_planner(
    planner: Any, *, max_candidates: int = 8
) -> List[Dict[str, Any]]:
    """Create an honest one-candidate FBE summary from currently materialized data.

    Fine-EQA's existing ``find_next_pose`` chooses and executes its own frontier
    internally. Until that planner is refactored to expose all scored frontiers,
    ``fbe_0`` represents that deterministic planner action. Features are only
    emitted when corresponding arrays are already available.
    """
    candidate: Dict[str, Any] = {"id": "fbe_0", "planner_action": "find_next_pose"}
    unexplored_neighbors = getattr(planner, "unexplored_neighbors", None)
    if unexplored_neighbors is not None:
        try:
            candidate["frontier_count"] = int((unexplored_neighbors > 0).sum())
        except Exception:
            pass
    unexplored = getattr(planner, "unexplored", None)
    if unexplored is not None:
        try:
            candidate["unexplored_fraction"] = float(unexplored.mean())
        except Exception:
            pass
    val_volume = getattr(planner, "_val_vol_cpu", None)
    if val_volume is not None:
        try:
            candidate["semantic_value_max"] = float(val_volume.max())
        except Exception:
            pass
    return [candidate][:max_candidates]


def current_fov_oracle_semantic(
    semantic_observation: Any,
    *,
    category_names: Optional[Mapping[int, str]] = None,
) -> Dict[str, Any]:
    """Summarize only labels visible in the current semantic sensor frame.

    The caller must pass the current observation, not a full scene semantic
    graph. ``category_names`` is optional and should be the sensor's local
    category mapping; absent names are represented by their numeric ids.
    """
    if semantic_observation is None:
        raise ValueError("oracle semantic mode requires a current semantic observation")
    try:
        values = semantic_observation.ravel().tolist()
    except AttributeError:
        values = []

        def _flatten(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for child in value:
                    _flatten(child)
            else:
                values.append(value)

        _flatten(semantic_observation)
    unique = []
    seen = set()
    for value in values:
        try:
            key = int(value)
        except (TypeError, ValueError):
            key = str(value)
        if key in seen or key in {-1, 0}:
            continue
        seen.add(key)
        unique.append(category_names.get(key, key) if category_names else key)
    return {
        "semantic_source": "habitat_current_fov",
        "current_fov_labels": unique,
    }
