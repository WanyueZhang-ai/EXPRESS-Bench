"""Online construction helpers used by the EXPRESS-Bench Habitat loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .answer_backend import AnswerBackend, AnswerResult
from .decision_router import (
    DEFAULT_DIRECT_ACTIONS,
    InternDecisionRouter,
    RouterDecision,
    decision_to_trace,
)
from .evidence_adapters import build_evidence_state
from .intern_decision_engine import InternDecisionEngine


def _get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    try:
        return config[key] if key in config else default
    except (TypeError, KeyError):
        return default


def _mapping(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    try:
        return {key: config[key] for key in config.keys()}
    except (AttributeError, TypeError):
        return {}


@dataclass
class InternDecisionRuntime:
    config: dict[str, Any]
    engine: InternDecisionEngine
    router: InternDecisionRouter
    answer_backend: AnswerBackend

    @property
    def evidence_mode(self) -> str:
        return str(self.config.get("evidence_mode", "vlm_state"))

    @property
    def action_mode(self) -> str:
        return str(self.config.get("action_mode", "planner_candidates"))

    @property
    def answer_mode(self) -> str:
        return str(self.config.get("answer_backend", "qwen_vl"))

    def evidence(self, **kwargs: Any) -> dict[str, Any]:
        return build_evidence_state(mode=self.evidence_mode, **kwargs)

    def decide(self, state: Mapping[str, Any], *, baseline_action: str, baseline_target_id: Optional[str]) -> RouterDecision:
        return self.router.decide(
            state,
            baseline_action=baseline_action,
            baseline_target_id=baseline_target_id,
        )

    def answer(self, question: str, question_type: str, state: Mapping[str, Any]) -> AnswerResult:
        return self.answer_backend.answer(question, question_type, state)

    def trace(
        self,
        decision: RouterDecision,
        *,
        question_index: Any,
        step: int,
        seed: Any,
    ) -> dict[str, Any]:
        return decision_to_trace(
            decision,
            question_index=question_index,
            step=step,
            evidence_mode=self.evidence_mode,
            action_mode=self.action_mode,
            answer_backend=self.answer_mode,
            seed=seed,
        )


def runtime_from_config(
    config: Any,
    *,
    qwen_fn: Any = None,
) -> Optional[InternDecisionRuntime]:
    raw = _mapping(config)
    if not bool(raw.get("enabled", False)):
        return None
    evidence_mode = str(raw.get("evidence_mode", "vlm_state"))
    action_mode = str(raw.get("action_mode", "planner_candidates"))
    if evidence_mode not in {"vlm_state", "decision_image", "decision_objects"}:
        raise ValueError("intern_decision.evidence_mode is invalid")
    if action_mode not in {"planner_candidates", "direct_actions"}:
        raise ValueError("intern_decision.action_mode is invalid")
    if action_mode == "direct_actions" and evidence_mode == "vlm_state":
        raise ValueError("direct_actions requires decision_image or decision_objects evidence")
    direct_actions = tuple(raw.get("direct_action_whitelist", DEFAULT_DIRECT_ACTIONS))
    safe_action = str(raw.get("safe_direct_action", "turn_right"))
    engine = InternDecisionEngine(
        checkpoint=str(raw.get("checkpoint", "/ssd/wyzhang/all_models/Intern-Decision-4B")),
        device=str(raw.get("device", "cuda")),
        dtype=str(raw.get("dtype", "bfloat16")),
        temperature=float(raw.get("temperature", 1.99241824)),
        media_root=str(raw.get("media_root", ".")),
        max_length=int(raw.get("max_length", 8192)),
        attn_implementation=str(raw.get("attn_implementation", "sdpa")),
        inference_module=raw.get("inference_module"),
    )
    router = InternDecisionRouter(
        engine,
        stop_threshold=float(raw.get("stop_threshold", 0.75)),
        min_choice_confidence=float(raw.get("min_choice_confidence", 0.0)),
        max_candidates=int(raw.get("max_candidates", 8)),
        action_mode=action_mode,
        direct_actions=direct_actions,
        safe_direct_action=safe_action,
    )
    answer_backend = AnswerBackend(
        mode=str(raw.get("answer_backend", "qwen_vl")),
        engine=engine,
        answer_options=_mapping(raw.get("answer_options", {})),
        qwen_fn=qwen_fn,
    )
    return InternDecisionRuntime(raw, engine, router, answer_backend)
