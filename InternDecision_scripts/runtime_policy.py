"""Small runtime policies shared by the Habitat integration and tests."""
from __future__ import annotations


def use_vlm_planner_perception(
    intern_enabled: bool,
    action_mode: str,
    *,
    evidence_mode: str | None = None,
) -> bool:
    """Keep existing TSDF candidate generation for planner experiments."""
    del evidence_mode
    return (not intern_enabled) or action_mode == "planner_candidates"
