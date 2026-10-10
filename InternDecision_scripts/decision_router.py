"""Structured Intern-Decision routing for planner and direct-action modes."""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .evidence_adapters import request_with_questions
from .intern_decision_engine import InternDecisionError


DEFAULT_DIRECT_ACTIONS = ("move_forward", "turn_left", "turn_right", "stop")


@dataclass
class RouterConfig:
    stop_threshold: float = 0.75
    min_choice_confidence: float = 0.0
    max_candidates: int = 8
    action_mode: str = "planner_candidates"
    direct_actions: tuple[str, ...] = DEFAULT_DIRECT_ACTIONS
    safe_direct_action: str = "turn_right"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.stop_threshold) <= 1.0:
            raise ValueError("stop_threshold must be in [0, 1]")
        if not 0.0 <= float(self.min_choice_confidence) <= 1.0:
            raise ValueError("min_choice_confidence must be in [0, 1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.action_mode not in {"planner_candidates", "direct_actions"}:
            raise ValueError("action_mode must be planner_candidates or direct_actions")
        self.direct_actions = tuple(str(action) for action in self.direct_actions)
        if not self.direct_actions or len(set(self.direct_actions)) != len(self.direct_actions):
            raise ValueError("direct_actions must be a non-empty unique sequence")
        if self.safe_direct_action not in self.direct_actions:
            raise ValueError("safe_direct_action must be in direct_actions")


@dataclass
class RouterDecision:
    action: str
    target_id: Optional[str]
    fallback: bool
    fallback_reason: Optional[str] = None
    stop_probability: Optional[float] = None
    mode_probability: Optional[float] = None
    target_probability: Optional[float] = None
    action_probability: Optional[float] = None
    response_model: Optional[str] = None
    latency_ms: Optional[float] = None
    usage: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    state: Optional[dict[str, Any]] = None
    response: Optional[dict[str, Any]] = None


class InternDecisionRouter:
    """Validate one structured model response and return an executable choice."""

    def __init__(
        self,
        engine: Any,
        *,
        stop_threshold: float = 0.75,
        min_choice_confidence: float = 0.0,
        max_candidates: int = 8,
        action_mode: str = "planner_candidates",
        direct_actions: Sequence[str] = DEFAULT_DIRECT_ACTIONS,
        safe_direct_action: str = "turn_right",
    ) -> None:
        self.engine = engine
        self.config = RouterConfig(
            stop_threshold=stop_threshold,
            min_choice_confidence=min_choice_confidence,
            max_candidates=max_candidates,
            action_mode=action_mode,
            direct_actions=tuple(direct_actions),
            safe_direct_action=safe_direct_action,
        )

    @staticmethod
    def _answer(response: Mapping[str, Any], field: str) -> Optional[Mapping[str, Any]]:
        answers = response.get("answers")
        value = answers.get(field) if isinstance(answers, Mapping) else None
        return value if isinstance(value, Mapping) else None

    @staticmethod
    def _choice(answer: Mapping[str, Any]) -> Optional[str]:
        choice = answer.get("choice", answer.get("decision"))
        return str(choice) if isinstance(choice, str) else None

    @staticmethod
    def _probability(answer: Mapping[str, Any], selected: str) -> Optional[float]:
        probabilities = answer.get("probabilities")
        value = probabilities.get(selected) if isinstance(probabilities, Mapping) else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None

    @staticmethod
    def _confidence(answer: Mapping[str, Any]) -> Optional[float]:
        try:
            value = float(answer.get("confidence"))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None

    def _questions(self, state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        questions: dict[str, dict[str, Any]] = {
            "stop": {
                "type": "noul",
                "instructions": "Is the current evidence sufficient to answer the question now?",
                "criteria": {
                    "no": "More exploration or evidence is needed.",
                    "yes": "The current evidence is sufficient; stop and answer.",
                },
            }
        }
        if self.config.action_mode == "direct_actions":
            questions["action"] = {
                "type": "choice",
                "instructions": "Which single valid Habitat action should run next?",
                "criteria": {
                    action: "Execute one transition named " + action
                    for action in self.config.direct_actions
                },
            }
            return questions
        map_state = state.get("map") if isinstance(state, Mapping) else None
        map_state = map_state if isinstance(map_state, Mapping) else {}
        goe = [item for item in map_state.get("goe_candidates", []) if isinstance(item, Mapping)]
        fbe = [item for item in map_state.get("fbe_candidates", []) if isinstance(item, Mapping)]
        modes: dict[str, str] = {}
        if goe:
            modes["goe"] = "Navigate toward a question-relevant semantic region candidate."
        if fbe:
            modes["fbe"] = "Navigate toward an unexplored frontier candidate."
        if modes:
            questions["mode"] = {
                "type": "choice",
                "instructions": "Which executable exploration mode should run next?",
                "criteria": modes,
            }
        candidates = goe + fbe
        if candidates:
            questions["target"] = {
                "type": "choice",
                "instructions": "Which currently feasible candidate should the executor target?",
                "criteria": {
                    str(item["id"]): json.dumps(
                        {key: value for key, value in item.items() if key != "id"},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    for item in candidates
                    if isinstance(item.get("id"), str)
                },
            }
        return questions

    def _fallback(
        self,
        *,
        state: Mapping[str, Any],
        action: str,
        target_id: Optional[str],
        reason: str,
        response: Optional[Mapping[str, Any]] = None,
    ) -> RouterDecision:
        usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
        timing = response.get("timing", {}) if isinstance(response, Mapping) else {}
        return RouterDecision(
            action=action,
            target_id=target_id,
            fallback=True,
            fallback_reason=reason,
            response_model=response.get("model") if isinstance(response, Mapping) else None,
            latency_ms=timing.get("inference_ms") if isinstance(timing, Mapping) else None,
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            status="fallback",
            state=copy.deepcopy(dict(state)),
            response=copy.deepcopy(dict(response)) if isinstance(response, Mapping) else None,
        )

    def decide(
        self,
        state: Mapping[str, Any],
        *,
        baseline_action: str,
        baseline_target_id: Optional[str],
    ) -> RouterDecision:
        if not isinstance(state, Mapping):
            raise ValueError("state must be a mapping")
        if self.config.action_mode == "direct_actions":
            baseline_target_id = None
            baseline_action = baseline_action or self.config.safe_direct_action
        questions = self._questions(state)
        request = request_with_questions(state, questions)
        started = time.perf_counter()
        try:
            response = self.engine.predict(request)
        except Exception as exc:
            reason = "model_error" if isinstance(exc, InternDecisionError) else "unexpected_model_error"
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id, reason=reason)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not isinstance(response, Mapping):
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id, reason="invalid_response")
        stop = self._answer(response, "stop")
        if stop is None or stop.get("type") != "noul":
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id, reason="invalid_stop", response=response)
        try:
            stop_probability = float(stop["noul"])
        except (KeyError, TypeError, ValueError):
            stop_probability = math.nan
        if not math.isfinite(stop_probability) or not 0.0 <= stop_probability <= 1.0:
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id, reason="invalid_stop", response=response)
        if stop_probability >= self.config.stop_threshold:
            decision = RouterDecision(
                action="stop", target_id=None, fallback=False,
                stop_probability=stop_probability,
                response_model=response.get("model"),
                latency_ms=self._latency(response, elapsed),
                usage=self._usage(response), state=copy.deepcopy(dict(state)),
                response=copy.deepcopy(dict(response)),
            )
            return decision

        if self.config.action_mode == "direct_actions":
            action_answer = self._answer(response, "action")
            selected = self._valid_choice(action_answer, self.config.direct_actions)
            if selected is None:
                return self._fallback(state=state, action=baseline_action, target_id=None, reason="invalid_action", response=response)
            confidence = self._confidence(action_answer or {})
            if confidence is None or confidence < self.config.min_choice_confidence:
                return self._fallback(state=state, action=baseline_action, target_id=None, reason="low_action_confidence", response=response)
            return RouterDecision(
                action=selected, target_id=None, fallback=False,
                stop_probability=stop_probability,
                action_probability=self._probability(action_answer or {}, selected),
                response_model=response.get("model"), latency_ms=self._latency(response, elapsed),
                usage=self._usage(response), state=copy.deepcopy(dict(state)), response=copy.deepcopy(dict(response)),
            )

        mode_answer = self._answer(response, "mode")
        map_state = state.get("map", {}) if isinstance(state.get("map", {}), Mapping) else {}
        available = {
            "goe": bool(map_state.get("goe_candidates")),
            "fbe": bool(map_state.get("fbe_candidates")),
        }
        mode = self._valid_choice(mode_answer, tuple(key for key, present in available.items() if present))
        if mode is None:
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id, reason="invalid_mode", response=response)
        confidence = self._confidence(mode_answer or {})
        if confidence is None or confidence < self.config.min_choice_confidence:
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id, reason="low_mode_confidence", response=response)
        target_answer = self._answer(response, "target")
        candidates = [item for key in ("goe_candidates", "fbe_candidates") for item in map_state.get(key, []) if isinstance(item, Mapping)]
        valid_ids = [str(item.get("id")) for item in candidates if item.get("id") is not None and str(item.get("id")).startswith(mode + "_")]
        target_id = self._valid_choice(target_answer, valid_ids)
        if target_id is None:
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id, reason="invalid_target", response=response)
        target_conf = self._confidence(target_answer or {})
        if target_conf is None or target_conf < self.config.min_choice_confidence:
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id, reason="low_target_confidence", response=response)
        return RouterDecision(
            action=mode, target_id=target_id, fallback=False,
            stop_probability=stop_probability,
            mode_probability=self._probability(mode_answer or {}, mode),
            target_probability=self._probability(target_answer or {}, target_id),
            response_model=response.get("model"), latency_ms=self._latency(response, elapsed),
            usage=self._usage(response), state=copy.deepcopy(dict(state)), response=copy.deepcopy(dict(response)),
        )

    @staticmethod
    def _valid_choice(answer: Optional[Mapping[str, Any]], allowed: Sequence[str]) -> Optional[str]:
        if answer is None or answer.get("type") != "choice":
            return None
        selected = InternDecisionRouter._choice(answer)
        return selected if selected in set(allowed) else None

    @staticmethod
    def _usage(response: Mapping[str, Any]) -> dict[str, Any]:
        usage = response.get("usage", {})
        return dict(usage) if isinstance(usage, Mapping) else {}

    @staticmethod
    def _latency(response: Mapping[str, Any], elapsed: float) -> float:
        timing = response.get("timing", {})
        value = timing.get("inference_ms") if isinstance(timing, Mapping) else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = elapsed
        return value if math.isfinite(value) else elapsed


def decision_to_trace(
    decision: RouterDecision,
    *,
    question_index: Any = None,
    step: Any = None,
    evidence_mode: str = "vlm_state",
    action_mode: str = "planner_candidates",
    answer_backend: str = "qwen_vl",
    seed: Any = None,
) -> dict[str, Any]:
    state = decision.state if isinstance(decision.state, Mapping) else {}
    map_state = state.get("map", {}) if isinstance(state.get("map", {}), Mapping) else {}
    candidate_ids = {
        key: [item.get("id") for item in map_state.get(key + "_candidates", []) if isinstance(item, Mapping)]
        for key in ("goe", "fbe")
    }
    evidence_summary = {}
    if isinstance(state.get("objects"), list):
        evidence_summary["object_count"] = len(state["objects"])
    if isinstance(state.get("images"), list):
        evidence_summary["image_count"] = len(state["images"])
    usage = decision.usage if isinstance(decision.usage, Mapping) else {}
    return {
        "schema_version": "express-bench-intern-decision-v1",
        "question_index": question_index,
        "step": step,
        "evidence_mode": evidence_mode,
        "action_mode": action_mode,
        "answer_backend": answer_backend,
        "seed": seed,
        "action": decision.action,
        "target_id": decision.target_id,
        "candidate_ids": candidate_ids,
        "fallback": bool(decision.fallback),
        "fallback_reason": decision.fallback_reason,
        "status": decision.status,
        "stop_probability": decision.stop_probability,
        "mode_probability": decision.mode_probability,
        "target_probability": decision.target_probability,
        "action_probability": decision.action_probability,
        "response_model": decision.response_model,
        "latency_ms": decision.latency_ms,
        "usage": dict(usage),
        "evidence_summary": evidence_summary,
    }
