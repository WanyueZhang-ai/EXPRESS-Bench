"""Build a compact, fair RGB and exploration-history state.

The encoder deliberately owns only online-observable state.  It is used by the
active-perception path and is independent from the legacy Fine-EQA VLM state.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORBIDDEN_KEYS = {
    "answer", "answers", "gold", "ground_truth", "goal", "goal_position",
    "target_pose", "target_position", "reference", "reference_answer",
    "reference_trajectory", "geodesic_distance", "trajectory",
}


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _safe(value: Any, path: str = "state") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            text = str(key)
            if text.lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden state field: {path}.{text}")
            result[text] = _safe(child, f"{path}.{text}")
        return result
    if isinstance(value, (list, tuple)):
        return [_safe(child, f"{path}[]") for child in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite state field: {path}")
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _safe(tolist(), path)
    item = getattr(value, "item", None)
    if callable(item):
        return _safe(item(), path)
    raise ValueError(f"unsupported state value at {path}: {type(value).__name__}")


@dataclass(frozen=True)
class ObservationRecord:
    step: int
    image_path: str
    digest: str
    pose_bin: tuple[int, ...]
    action: str | None
    displacement_m: float
    newly_explored_fraction: float
    collided: bool
    novelty: float


@dataclass
class ExplorationMemory:
    history_window: int = 8
    max_history_images: int = 4
    pose_bin_size_m: float = 0.5
    progress_threshold: float = 0.01
    records: list[ObservationRecord] = field(default_factory=list)
    no_progress_steps: int = 0

    def __post_init__(self) -> None:
        if self.history_window < 1 or self.max_history_images < 1:
            raise ValueError("history_window and max_history_images must be positive")
        if self.pose_bin_size_m <= 0 or not math.isfinite(float(self.pose_bin_size_m)):
            raise ValueError("pose_bin_size_m must be positive and finite")
        if self.progress_threshold < 0 or not math.isfinite(float(self.progress_threshold)):
            raise ValueError("progress_threshold must be non-negative and finite")

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def append(
        self,
        *,
        step: int,
        image_path: str | Path,
        pose: Sequence[float],
        action: str | None = None,
        displacement_m: float = 0.0,
        newly_explored_fraction: float = 0.0,
        collided: bool = False,
    ) -> ObservationRecord:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"image_path is not a readable file: {path}")
        try:
            coords = tuple(_finite_float(item, "pose") for item in pose)
        except TypeError as exc:
            raise ValueError("pose must be a finite sequence") from exc
        if not coords:
            raise ValueError("pose must not be empty")
        displacement = _finite_float(displacement_m, "displacement_m")
        explored = _finite_float(newly_explored_fraction, "newly_explored_fraction")
        if explored < 0:
            raise ValueError("newly_explored_fraction must be non-negative")
        digest = self._digest(path)
        pose_bin = tuple(int(round(value / self.pose_bin_size_m)) for value in coords)
        previous = self.records[-1] if self.records else None
        novelty = 0.0 if previous and previous.digest == digest and previous.pose_bin == pose_bin else 1.0
        progressed = explored >= self.progress_threshold
        self.no_progress_steps = 0 if progressed else self.no_progress_steps + 1
        record = ObservationRecord(
            step=int(step), image_path=str(path), digest=digest, pose_bin=pose_bin,
            action=str(action) if action is not None else None,
            displacement_m=displacement, newly_explored_fraction=explored,
            collided=bool(collided), novelty=novelty,
        )
        self.records.append(record)
        self.records = self.records[-self.history_window :]
        return record

    def image_paths(self) -> list[str]:
        if not self.records:
            return []
        current = self.records[-1].image_path
        selected: list[str] = [current]
        for record in reversed(self.records[:-1]):
            if record.image_path not in selected:
                selected.append(record.image_path)
            if len(selected) - 1 >= self.max_history_images:
                break
        return selected

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [record.step for record in self.records],
            "actions": [record.action for record in self.records if record.action is not None],
            "novelty": [record.novelty for record in self.records],
            "newly_explored_fraction": [record.newly_explored_fraction for record in self.records],
            "displacement_m": [record.displacement_m for record in self.records],
            "collisions": [record.collided for record in self.records],
            "last_collision": bool(self.records[-1].collided) if self.records else False,
            "no_progress_steps": self.no_progress_steps,
        }


class EvidenceStateEncoder:
    """Create immutable request state from the online RGB/history snapshot."""

    def __init__(
        self,
        question: str,
        question_type: str,
        *,
        history_window: int = 8,
        max_history_images: int = 4,
        pose_bin_size_m: float = 0.5,
        progress_threshold: float = 0.01,
    ) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        self.question = question
        self.question_type = str(question_type)
        self.memory = ExplorationMemory(
            history_window=history_window, max_history_images=max_history_images,
            pose_bin_size_m=pose_bin_size_m, progress_threshold=progress_threshold,
        )
        self._candidates: list[dict[str, Any]] = []

    def set_candidates(self, candidates: Iterable[Mapping[str, Any]] | None) -> None:
        normalized = []
        for candidate in candidates or []:
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate must be a mapping")
            normalized.append(_safe(dict(candidate), "candidates[]"))
        self._candidates = normalized

    def observe(self, **kwargs: Any) -> dict[str, Any]:
        record = self.memory.append(**kwargs)
        state = {
            "schema_version": "express-bench-evidence-aware-v1",
            "task": {"question": self.question, "question_type": self.question_type},
            "observation": {
                "step": record.step,
                "view_novelty": record.novelty,
                "newly_explored_fraction": record.newly_explored_fraction,
                "displacement_m": record.displacement_m,
                "collided": record.collided,
            },
            "history": self.memory.as_dict(),
            "images": self.memory.image_paths(),
            "candidates": list(self._candidates),
        }
        return _safe(state)

    def state_json(self, state: Mapping[str, Any]) -> str:
        return json.dumps(_safe(state), ensure_ascii=False, sort_keys=True)
