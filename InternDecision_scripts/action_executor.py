"""Small, testable executor for direct atomic Habitat actions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass
class ActionResult:
    requested_action: str
    action: str
    stepped: bool
    fallback: bool = False
    collided: bool = False
    observation: Any = None
    fallback_reason: str | None = None


def path_increment(
    action: str,
    before: Sequence[float],
    after: Sequence[float],
    *,
    stepped: bool,
) -> float:
    """Return traveled distance for one direct action transition."""
    if not stepped or str(action) != "move_forward":
        return 0.0
    delta = [float(end) - float(start) for start, end in zip(before, after)]
    if len(delta) == 0:
        return 0.0
    return math.sqrt(sum(value * value for value in delta))


class DirectActionExecutor:
    """Map one validated model choice to at most one environment transition."""

    def __init__(
        self,
        *,
        action_space: Mapping[str, Any] | Sequence[str] | set[str],
        action_whitelist: Sequence[str] | None = None,
        safe_action: str = "turn_right",
    ) -> None:
        if isinstance(action_space, Mapping):
            names = set(str(key) for key in action_space)
        else:
            names = set(str(key) for key in action_space)
        self.action_space = names
        self.action_whitelist = tuple(action_whitelist or sorted(names))
        if "stop" not in self.action_whitelist:
            self.action_whitelist = tuple(self.action_whitelist) + ("stop",)
        self.safe_action = str(safe_action)
        required = set(self.action_whitelist) - {"stop"}
        missing = required - names
        if self.safe_action not in names:
            missing.add(self.safe_action)
        if missing:
            raise ValueError(f"configured direct actions are unavailable in Habitat action space: {sorted(missing)}")

    def execute(self, requested_action: str, step_fn: Callable[[str], Any]) -> ActionResult:
        requested = str(requested_action)
        if requested == "stop":
            return ActionResult(requested, "stop", stepped=False)
        action = requested
        fallback = False
        reason = None
        if action not in self.action_whitelist or action not in self.action_space:
            action = self.safe_action
            fallback = True
            reason = "invalid_action"
        observation = step_fn(action)
        collided = self._collision(observation)
        return ActionResult(
            requested_action=requested,
            action=action,
            stepped=True,
            fallback=fallback,
            collided=collided,
            observation=observation,
            fallback_reason=reason,
        )

    def safe_action_for_next_cycle(self, result: ActionResult) -> str | None:
        if result.collided and result.action == "move_forward":
            return self.safe_action
        return None

    @staticmethod
    def _collision(observation: Any) -> bool:
        if isinstance(observation, Mapping):
            return bool(observation.get("collided", observation.get("did_collide", False)))
        return bool(getattr(observation, "collided", getattr(observation, "did_collide", False)))
