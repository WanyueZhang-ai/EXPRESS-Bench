"""Optional online integration helpers used by ``main.py``."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from .decision_router import DecisionRouter, RouterConfig, RouterDecision, decision_to_trace
from .jev_client import JevClient, JevError
from .tsdf_candidates import current_fov_oracle_semantic


class _CallBudgetClient:
    def __init__(self, client: Any, max_calls: int) -> None:
        self.client = client
        self.max_calls = int(max_calls)
        self.calls = 0

    def evaluate(self, state: Any, questions: Mapping[str, Any]) -> Any:
        if self.calls >= self.max_calls:
            raise JevError("jev_call_budget_exceeded")
        self.calls += 1
        return self.client.evaluate(state, questions)


def _get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    try:
        return config[key] if key in config else default
    except (TypeError, KeyError):
        return default


def router_from_config(config: Any) -> Optional[DecisionRouter]:
    """Build an online router only when the explicit config flag is enabled."""
    if not bool(_get(config, "enabled", False)):
        return None
    mode = str(_get(config, "mode", "fair"))
    client = JevClient(
        api_key=_get(config, "api_key", None),
        model=str(_get(config, "model", "jev-latest")),
        timeout=float(_get(config, "timeout_seconds", 20.0)),
        max_retries=int(_get(config, "max_retries", 2)),
    )
    client = _CallBudgetClient(client, int(_get(config, "max_calls", 100000)))
    return DecisionRouter(
        client,
        RouterConfig(
            stop_threshold=float(_get(config, "stop_threshold", 0.75)),
            min_choice_confidence=float(_get(config, "min_choice_confidence", 0.0)),
            max_candidates=int(_get(config, "max_candidates", 8)),
            evidence_mode=mode,
        ),
    )


def baseline_route(exp_fbe_state: bool, candidate_ids: Sequence[str]) -> Tuple[str, Optional[str]]:
    mode = "fbe" if exp_fbe_state else "goe"
    target = next((candidate for candidate in candidate_ids if candidate.startswith(mode + "_")), None)
    return mode, target


def target_index(target_id: Optional[str], prefix: str) -> Optional[int]:
    if not isinstance(target_id, str) or not target_id.startswith(prefix + "_"):
        return None
    try:
        index = int(target_id[len(prefix) + 1 :])
    except ValueError:
        return None
    return index if index >= 0 else None


def oracle_evidence_from_observation(
    semantic_observation: Any, *, category_names: Optional[Mapping[int, str]] = None
) -> Mapping[str, Any]:
    """Fail loudly unless a current semantic sensor frame is provided."""
    return current_fov_oracle_semantic(semantic_observation, category_names=category_names)


def trace_for_main(
    decision: RouterDecision, *, episode_id: Any, step: int, seed: Any, mode: str
) -> Mapping[str, Any]:
    return decision_to_trace(
        decision,
        episode_id=episode_id,
        step=step,
        seed=seed,
        router_mode=mode,
    )


def semantic_category_names(simulator: Any) -> Mapping[int, str]:
    """Build a local id->category map without sending the scene graph to Jev."""
    mapping = {}
    scene = getattr(simulator, "semantic_scene", None)
    objects = getattr(scene, "objects", None) if scene is not None else None
    if objects is None:
        return mapping
    for index, obj in enumerate(objects):
        category = getattr(obj, "category", None)
        name = None
        try:
            name = category.name() if callable(getattr(category, "name", None)) else getattr(category, "name", None)
        except Exception:
            name = None
        if not name:
            continue
        mapping[index] = str(name)
        mapping[index + 1] = str(name)
        object_id = getattr(obj, "id", None)
        if isinstance(object_id, str):
            try:
                mapping[int(object_id.rsplit("_", 1)[-1])] = str(name)
            except (ValueError, IndexError):
                pass
    return mapping
