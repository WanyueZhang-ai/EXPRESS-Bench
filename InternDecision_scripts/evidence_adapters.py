"""Build safe, same-timestep evidence for Intern-Decision requests."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


EVIDENCE_MODES = {"vlm_state", "decision_image", "decision_objects"}
_FORBIDDEN_KEYS = {
    "answer", "answers", "goal", "goal_position", "goal_rotation",
    "geodesic_distance", "reference", "reference_answer",
    "reference_trajectory", "trajectory", "ground_truth", "full_scene_labels",
    "future_labels", "target_pose", "target_position", "raw_goal",
}
_CANDIDATE_FIELDS = {
    "region", "region_id", "direction", "count", "point_count",
    "point_available", "planner_action", "frontier_count", "unexplored_fraction",
    "semantic_value_max", "relative_direction", "relative_distance_bin",
}


class EvidenceError(ValueError):
    """Evidence cannot be represented without privileged or unsafe data."""


def _json_value(value: Any, path: str = "state") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            text = str(key)
            if text.lower() in _FORBIDDEN_KEYS:
                raise EvidenceError(f"forbidden evidence field at {path}.{text}")
            result[text] = _json_value(child, f"{path}.{text}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(child, f"{path}[]") for child in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError(f"non-finite evidence value at {path}")
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item(), path)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist(), path)
    raise EvidenceError(f"non-JSON evidence value at {path}: {type(value).__name__}")


def _flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _flatten(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten(child)
    else:
        # Habitat's semantic sensor returns a NumPy array. Convert array-like
        # values before flattening so each visible id is inspected separately.
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                converted = tolist()
            except Exception:
                converted = None
            if converted is not None and converted is not value:
                yield from _flatten(converted)
                return
        yield value


def _visible_ids(semantic_observation: Any) -> set[int]:
    result: set[int] = set()
    if semantic_observation is None:
        return result
    for value in _flatten(semantic_observation):
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number not in {-1, 0}:
            result.add(number)
    return result


def _object_id(obj: Any, index: int) -> str:
    value = getattr(obj, "id", None)
    return str(value) if value is not None else f"object_{index}"


def _object_numeric_ids(object_id: str, index: int) -> set[int]:
    result = {index, index + 1}
    try:
        result.add(int(object_id.rsplit("_", 1)[-1]))
    except (TypeError, ValueError, IndexError):
        pass
    return result


def _category_name(obj: Any) -> str:
    category = getattr(obj, "category", None)
    name = getattr(category, "name", None)
    try:
        value = name() if callable(name) else name
    except Exception:
        value = None
    return str(value) if value else "unknown"


def _vector(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    try:
        rows = list(value)
    except TypeError:
        return None
    try:
        result = [float(x) for x in rows]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(x) for x in result) else None


def visible_object_summaries(
    semantic_observation: Any,
    scene_objects: Sequence[Any],
    *,
    agent_position: Optional[Sequence[float]] = None,
) -> list[dict[str, Any]]:
    """Serialize only objects whose semantic ids occur in the current frame."""
    visible = _visible_ids(semantic_observation)
    agent = _vector(agent_position)
    result: list[dict[str, Any]] = []
    for index, obj in enumerate(scene_objects or []):
        object_id = _object_id(obj, index)
        if visible and not (_object_numeric_ids(object_id, index) & visible):
            continue
        if not visible:
            continue
        center = _vector(getattr(obj, "translation", None))
        if center is None:
            center = _vector(getattr(obj, "center", None))
        aabb = getattr(obj, "aabb", None)
        if center is None and aabb is not None:
            center = _vector(getattr(aabb, "center", None))
        extent = None
        if aabb is not None:
            extent = _vector(getattr(aabb, "sizes", None))
            if extent is None:
                extent = _vector(getattr(aabb, "size", None))
        row: dict[str, Any] = {"id": object_id, "category": _category_name(obj)}
        if center is not None:
            row["center_m"] = center
            if agent is not None and len(agent) == len(center):
                row["agent_relative_center_m"] = [round(x - y, 6) for x, y in zip(center, agent)]
        if extent is not None:
            row["extent_m"] = extent
        result.append(row)
    return result


def normalize_candidates(
    candidates: Iterable[Mapping[str, Any]] | None,
    prefix: str,
    *,
    max_candidates: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    rows: list[dict[str, Any]] = []
    index_map: dict[str, int] = {}
    for source_index, candidate in enumerate(candidates or []):
        if not isinstance(candidate, Mapping):
            raise EvidenceError(f"{prefix} candidate must be a mapping")
        row: dict[str, Any] = {"id": f"{prefix}_{len(rows)}"}
        for key in _CANDIDATE_FIELDS:
            if key in candidate:
                row[key] = _json_value(candidate[key], f"{prefix}[{source_index}].{key}")
        rows.append(row)
        index_map[row["id"]] = source_index
        if len(rows) >= max_candidates:
            break
    return rows, index_map


def _safe_history(history: Mapping[str, Any] | None) -> dict[str, Any]:
    if history is None:
        return {}
    if not isinstance(history, Mapping):
        raise EvidenceError("history must be a mapping")
    return _json_value(history, "history")


def build_evidence_state(
    *,
    mode: str,
    question: str,
    question_type: str,
    history: Mapping[str, Any] | None,
    perception: Mapping[str, Any] | None,
    goe_candidates: Iterable[Mapping[str, Any]] | None,
    fbe_candidates: Iterable[Mapping[str, Any]] | None,
    max_candidates: int = 8,
    image_path: str | Path | None = None,
    objects: Sequence[Mapping[str, Any]] | None = None,
    semantic_observation: Any = None,
    scene_objects: Sequence[Any] | None = None,
    agent_position: Optional[Sequence[float]] = None,
    step: int | None = None,
    decision_id: str | None = None,
) -> dict[str, Any]:
    if mode not in EVIDENCE_MODES:
        raise EvidenceError(f"unsupported evidence mode: {mode}")
    if not isinstance(question, str) or not question.strip():
        raise EvidenceError("question must be a non-empty string")
    goe, goe_index = normalize_candidates(goe_candidates, "goe", max_candidates=max_candidates)
    fbe, fbe_index = normalize_candidates(fbe_candidates, "fbe", max_candidates=max_candidates)
    safe_perception = _json_value(perception or {}, "perception")
    state: dict[str, Any] = {
        "schema_version": "express-bench-intern-decision-v1",
        "evidence_mode": mode,
        "task": {"question": question, "question_type": str(question_type)},
        "history": _safe_history(history),
        "perception": safe_perception,
        "map": {"goe_candidates": goe, "fbe_candidates": fbe},
        "candidate_index": {"goe": goe_index, "fbe": fbe_index},
    }
    if step is not None:
        state["step"] = int(step)
    if decision_id is not None:
        state["decision_id"] = str(decision_id)
    if mode == "decision_image":
        if image_path is None:
            raise EvidenceError("decision_image mode requires image_path")
        path = Path(image_path).expanduser().resolve()
        if not path.is_file() or not path.stat().st_size:
            raise EvidenceError(f"decision image is not a readable file: {path}")
        state["images"] = [str(path)]
    elif mode == "decision_objects":
        if objects is None and scene_objects is not None:
            objects = visible_object_summaries(
                semantic_observation, scene_objects, agent_position=agent_position
            )
        state["objects"] = _json_value(list(objects or []), "objects")
    # VLM state intentionally contains no images or object graph.
    json.dumps(state, ensure_ascii=False)
    return state


def request_with_questions(state: Mapping[str, Any], questions: Mapping[str, Any]) -> dict[str, Any]:
    """Create a model request without mutating the evidence snapshot."""
    request = copy.deepcopy(dict(state))
    images = request.pop("images", None)
    result = {"state": request, "questions": copy.deepcopy(dict(questions))}
    if images:
        result["images"] = images
    return result
