"""Compose evidence memory, one-forward scores, and local exploration policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence_backbone import BackboneScores, EvidenceAwareDecisionBackbone
from .evidence_policy import EvidenceAwarePolicy, PolicyDecision
from .evidence_state_encoder import EvidenceStateEncoder


@dataclass(frozen=True)
class ActiveStepResult:
    decision: PolicyDecision
    state: dict[str, Any]
    scores: BackboneScores | None
    trace: dict[str, Any]


class ActivePerceptionController:
    """Episode-local controller; each ``step`` issues at most one model call."""

    def __init__(
        self,
        question: str,
        question_type: str,
        *,
        engine: Any,
        max_candidates: int = 7,
        max_steps: int = 100,
        history_window: int = 8,
        max_history_images: int = 3,
        progress_threshold: float = 0.01,
        stop_threshold: float = 0.75,
        utility_threshold: float = 0.05,
        no_progress_patience: int = 3,
        cost_weight: float = 0.5,
        novelty_weight: float = 0.1,
    ) -> None:
        self.encoder = EvidenceStateEncoder(
            question, question_type, history_window=history_window,
            max_history_images=max_history_images,
            progress_threshold=progress_threshold,
        )
        self.backbone = EvidenceAwareDecisionBackbone(engine, max_candidates=max_candidates)
        self.policy = EvidenceAwarePolicy(
            stop_threshold=stop_threshold, utility_threshold=utility_threshold,
            no_progress_patience=no_progress_patience, cost_weight=cost_weight,
            novelty_weight=novelty_weight,
        )
        self.max_steps = int(max_steps)

    def step(
        self,
        *,
        step: int,
        image_path: str | Path,
        pose: Sequence[float],
        candidates: Sequence[Mapping[str, Any]],
        action: str | None = None,
        displacement_m: float = 0.0,
        newly_explored_fraction: float = 0.0,
        collided: bool = False,
    ) -> ActiveStepResult:
        self.encoder.set_candidates(candidates)
        state = self.encoder.observe(
            step=step, image_path=image_path, pose=pose, action=action,
            displacement_m=displacement_m,
            newly_explored_fraction=newly_explored_fraction,
            collided=collided,
        )
        scores = None
        error_type = None
        attempted_score = False
        if candidates and step < self.max_steps - 1 and self.encoder.memory.no_progress_steps < self.policy.no_progress_patience:
            attempted_score = True
            try:
                scores = self.backbone.score(state)
            except Exception as exc:
                error_type = exc.__class__.__name__
        decision = self.policy.decide(
            scores, candidates, step=step, max_steps=self.max_steps,
            no_progress_steps=self.encoder.memory.no_progress_steps,
        )
        trace = {
            "schema_version": "express-bench-evidence-aware-v1",
            "step": int(step),
            "candidate_ids": [str(row["id"]) for row in candidates],
            "action": decision.action,
            "target_id": decision.target_id,
            "reason": decision.reason,
            "fallback": decision.fallback,
            "error_type": error_type,
            "verifier": decision.verifier,
            "utility": decision.utility,
            "selected_value": decision.selected_value,
            "selected_cost": decision.selected_cost,
            "decision_fields": scores.decision_fields if scores else (self.backbone.last_decision_fields if attempted_score else 0),
            "model_calls": scores.model_calls if scores else int(attempted_score and self.backbone.last_predict_called),
            "latency_ms": scores.latency_ms if scores else None,
            "usage": scores.usage if scores else {},
            "candidate_scores": {
                identifier: {"value": scores.value[identifier], "cost": scores.cost[identifier]}
                for identifier in scores.value
            } if scores else {},
            "history": {
                "no_progress_steps": state["history"]["no_progress_steps"],
                "view_novelty": state["observation"]["view_novelty"],
                "newly_explored_fraction": state["observation"]["newly_explored_fraction"],
                "image_count": len(state["images"]),
            },
        }
        return ActiveStepResult(decision, state, scores, trace)
