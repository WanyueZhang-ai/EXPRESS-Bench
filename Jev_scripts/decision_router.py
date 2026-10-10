"""Pure Jev decision routing for the EXPRESS-Bench online loop."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .jev_client import (
    JevClient,
    JevError,
    JevHTTPError,
    JevResponse,
    JevSchemaError,
    JevTransportError,
)


SCHEMA_VERSION = "express-bench-jev-router-v1"
_FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "goal",
    "goal_position",
    "goal_rotation",
    "geodesic_distance",
    "reference",
    "reference_answer",
    "reference_trajectory",
    "trajectory",
    "ground_truth",
    "full_scene_labels",
    "future_labels",
}


@dataclass
class RouterConfig:
    stop_threshold: float = 0.75
    min_choice_confidence: float = 0.0
    max_candidates: int = 8
    evidence_mode: str = "fair"

    def __post_init__(self) -> None:
        if not 0.0 <= self.stop_threshold <= 1.0:
            raise ValueError("stop_threshold must be in [0, 1]")
        if not 0.0 <= self.min_choice_confidence <= 1.0:
            raise ValueError("min_choice_confidence must be in [0, 1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.evidence_mode not in {"fair", "oracle_semantic"}:
            raise ValueError("evidence_mode must be fair or oracle_semantic")


@dataclass
class RouterDecision:
    action: str
    target_id: Optional[str]
    fallback: bool
    fallback_reason: Optional[str] = None
    stop_probability: Optional[float] = None
    mode_probability: Optional[float] = None
    target_probability: Optional[float] = None
    response_model: Optional[str] = None
    latency_ms: Optional[float] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    status: str = "ok"
    state: Optional[Dict[str, Any]] = None


def _reject_forbidden(value: Any, path: str = "state") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_KEYS:
                raise ValueError("forbidden privileged field at {}.{}".format(path, key_text))
            _reject_forbidden(child, "{}.{}".format(path, key_text))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, "{}[{}]".format(path, index))


def _copy_json(value: Any) -> Any:
    """Copy only JSON-compatible values and fail clearly for arrays/objects."""
    try:
        copied = copy.deepcopy(value)
        json.dumps(copied, ensure_ascii=False)
        return copied
    except (TypeError, ValueError) as exc:
        raise ValueError("state contains a non-JSON value: {}".format(exc))


def _normalize_candidates(
    candidates: Iterable[Mapping[str, Any]], prefix: str, max_candidates: int
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates or []):
        if not isinstance(candidate, Mapping):
            raise ValueError("{} candidate must be an object".format(prefix))
        item = {str(key): _copy_json(value) for key, value in candidate.items()}
        candidate_id = item.get("id", "{}_{}".format(prefix, index))
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("{} candidate id must be a non-empty string".format(prefix))
        item["id"] = candidate_id
        normalized.append(item)
    # Preserve planner order. The same truncation is used for fair and oracle.
    return normalized[:max_candidates]


def build_state(
    *,
    question: str,
    question_type: str,
    history: Mapping[str, Any],
    perception: Mapping[str, Any],
    goe_candidates: Sequence[Mapping[str, Any]],
    fbe_candidates: Sequence[Mapping[str, Any]],
    evidence_mode: str = "fair",
    oracle_semantic: Optional[Mapping[str, Any]] = None,
    max_candidates: int = 8,
) -> Dict[str, Any]:
    """Build the only state shape that may be sent to Jev.

    ``oracle_semantic`` is deliberately restricted to current-FOV labels. The
    fair and oracle modes share all other fields and candidate IDs.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if evidence_mode not in {"fair", "oracle_semantic"}:
        raise ValueError("evidence_mode must be fair or oracle_semantic")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    _reject_forbidden(history)
    _reject_forbidden(perception)
    _reject_forbidden(goe_candidates)
    _reject_forbidden(fbe_candidates)
    if oracle_semantic is not None:
        _reject_forbidden(oracle_semantic)

    goe = _normalize_candidates(goe_candidates, "goe", max_candidates)
    fbe = _normalize_candidates(fbe_candidates, "fbe", max_candidates)
    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_mode": evidence_mode,
        "task": {"question": question, "question_type": str(question_type)},
        "history": _copy_json(history),
        "perception": _copy_json(perception),
        "map": {"goe_candidates": goe, "fbe_candidates": fbe},
    }
    if evidence_mode == "oracle_semantic":
        if not isinstance(oracle_semantic, Mapping):
            raise ValueError("oracle_semantic mode requires current-FOV semantic evidence")
        labels = oracle_semantic.get("current_fov_labels")
        source = oracle_semantic.get("semantic_source")
        if not isinstance(labels, list) or source != "habitat_current_fov":
            raise ValueError(
                "oracle semantic evidence must contain current_fov_labels and semantic_source=habitat_current_fov"
            )
        state["perception"]["oracle_semantic"] = _copy_json(oracle_semantic)
    elif oracle_semantic is not None:
        raise ValueError("oracle_semantic evidence cannot be supplied in fair mode")
    _reject_forbidden(state)
    return state


def _candidate_index(state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    map_state = state.get("map", {})
    for key in ("goe_candidates", "fbe_candidates"):
        for candidate in map_state.get(key, []):
            result[candidate["id"]] = candidate
    return result


def _baseline(
    *,
    action: str,
    target_id: Optional[str],
    state: Optional[Dict[str, Any]],
    reason: str,
    response: Optional[JevResponse] = None,
) -> RouterDecision:
    return RouterDecision(
        action=action,
        target_id=target_id,
        fallback=True,
        fallback_reason=reason,
        response_model=response.model if response else None,
        latency_ms=response.latency_ms if response else None,
        usage=response.usage if response else {},
        request_id=response.request_id if response else None,
        status="fallback",
        state=state,
    )


class DecisionRouter:
    """Ask Jev for stop/mode/target and return an executable decision."""

    def __init__(self, client: JevClient, config: Optional[RouterConfig] = None) -> None:
        self.client = client
        self.config = config or RouterConfig()

    def _questions(self, state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        goe = state["map"]["goe_candidates"]
        fbe = state["map"]["fbe_candidates"]
        questions: Dict[str, Dict[str, Any]] = {
            "stop": {
                "type": "noul",
                "instructions": "Given the question and current visual/map evidence, should the agent stop exploring and answer now?",
                "criteria": {
                    "true": "The current evidence is sufficient to answer the question.",
                    "false": "More exploration is needed or the evidence is insufficient.",
                },
            }
        }
        modes: Dict[str, str] = {}
        if goe:
            modes["goe"] = "Navigate toward a question-relevant semantic region candidate."
        if fbe:
            modes["fbe"] = "Navigate toward an unexplored frontier candidate."
        if modes:
            questions["mode"] = {
                "type": "choice",
                "instructions": "Which executable exploration mode should be used for the next step?",
                "criteria": modes,
            }
        candidates = list(goe) + list(fbe)
        if candidates:
            questions["target"] = {
                "type": "choice",
                "instructions": "Which currently feasible candidate id should the executor target?",
                "criteria": {
                    item["id"]: json.dumps(
                        {key: value for key, value in item.items() if key != "id"},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    for item in candidates
                },
            }
        return questions

    @staticmethod
    def _probability(answer: Mapping[str, Any], selected: str) -> Optional[float]:
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, Mapping) and selected in probabilities:
            try:
                return float(probabilities[selected])
            except (TypeError, ValueError):
                return None
        return None

    def decide(
        self,
        *,
        question: str,
        question_type: str,
        history: Mapping[str, Any],
        perception: Mapping[str, Any],
        goe_candidates: Sequence[Mapping[str, Any]],
        fbe_candidates: Sequence[Mapping[str, Any]],
        baseline_action: str,
        baseline_target_id: Optional[str],
        evidence_mode: Optional[str] = None,
        oracle_semantic: Optional[Mapping[str, Any]] = None,
    ) -> RouterDecision:
        if baseline_action not in {"goe", "fbe"}:
            raise ValueError("baseline_action must be goe or fbe")
        mode = evidence_mode or self.config.evidence_mode
        try:
            state = build_state(
                question=question,
                question_type=question_type,
                history=history,
                perception=perception,
                goe_candidates=goe_candidates,
                fbe_candidates=fbe_candidates,
                evidence_mode=mode,
                oracle_semantic=oracle_semantic,
                max_candidates=self.config.max_candidates,
            )
        except ValueError:
            # State construction errors are caller/configuration errors and must
            # not silently turn an oracle run into a fair run.
            raise
        questions = self._questions(state)
        try:
            response = self.client.evaluate(state, questions)
        except JevHTTPError as exc:
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="jev_http_{}".format(exc.status),
            )
        except JevSchemaError:
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="jev_schema_error",
            )
        except JevTransportError:
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="jev_transport_error",
            )
        except JevError as exc:
            reason = "call_budget_exceeded" if "call_budget" in str(exc) else "jev_error"
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason=reason,
            )
        except Exception:
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="unexpected_router_error",
            )

        stop = response.answer("stop")
        if not isinstance(stop, Mapping) or stop.get("type") != "noul":
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="invalid_stop",
                response=response,
            )
        try:
            stop_probability = float(stop["noul"])
        except (KeyError, TypeError, ValueError):
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="invalid_stop",
                response=response,
            )
        if stop_probability >= self.config.stop_threshold:
            return RouterDecision(
                action="stop",
                target_id=None,
                fallback=False,
                stop_probability=stop_probability,
                response_model=response.model,
                latency_ms=response.latency_ms,
                usage=response.usage,
                request_id=response.request_id,
                state=state,
            )

        mode_answer = response.answer("mode")
        target_answer = response.answer("target")
        candidates = _candidate_index(state)
        if not isinstance(mode_answer, Mapping) or mode_answer.get("type") != "choice":
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="invalid_mode",
                response=response,
            )
        selected_mode = mode_answer.get("choice")
        available_modes = {
            "goe": bool(state["map"]["goe_candidates"]),
            "fbe": bool(state["map"]["fbe_candidates"]),
        }
        if selected_mode not in available_modes or not available_modes[selected_mode]:
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="invalid_mode",
                response=response,
            )
        confidence = mode_answer.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < self.config.min_choice_confidence:
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="low_mode_confidence",
                response=response,
            )

        if not isinstance(target_answer, Mapping) or target_answer.get("type") != "choice":
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="invalid_target",
                response=response,
            )
        target_id = target_answer.get("choice")
        target = candidates.get(target_id) if isinstance(target_id, str) else None
        if target is None or not target_id.startswith(selected_mode + "_"):
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="invalid_target",
                response=response,
            )
        target_confidence = target_answer.get("confidence", 1.0)
        try:
            target_confidence = float(target_confidence)
        except (TypeError, ValueError):
            target_confidence = -1.0
        if target_confidence < self.config.min_choice_confidence:
            return _baseline(
                action=baseline_action,
                target_id=baseline_target_id,
                state=state,
                reason="low_target_confidence",
                response=response,
            )
        return RouterDecision(
            action=selected_mode,
            target_id=target_id,
            fallback=False,
            stop_probability=stop_probability,
            mode_probability=self._probability(mode_answer, selected_mode),
            target_probability=self._probability(target_answer, target_id),
            response_model=response.model,
            latency_ms=response.latency_ms,
            usage=response.usage,
            request_id=response.request_id,
            state=state,
        )


def decision_to_trace(
    decision: RouterDecision,
    *,
    episode_id: Any = None,
    step: Any = None,
    seed: Any = None,
    router_mode: str = "fair",
) -> Dict[str, Any]:
    """Return a durable, secret-free trace record for one online decision."""
    usage = decision.usage if isinstance(decision.usage, Mapping) else {}
    map_state = decision.state.get("map", {}) if isinstance(decision.state, Mapping) else {}
    candidate_ids = {
        "goe": [item.get("id") for item in map_state.get("goe_candidates", [])],
        "fbe": [item.get("id") for item in map_state.get("fbe_candidates", [])],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": step,
        "seed": seed,
        "router_mode": router_mode,
        "semantic_source": "habitat_current_fov" if router_mode == "oracle_semantic" else "vlm_tsdf",
        "action": decision.action,
        "target_id": decision.target_id,
        "candidate_ids": candidate_ids,
        "fallback": bool(decision.fallback),
        "fallback_reason": decision.fallback_reason,
        "status": decision.status,
        "stop_probability": decision.stop_probability,
        "mode_probability": decision.mode_probability,
        "target_probability": decision.target_probability,
        "response_model": decision.response_model,
        "latency_ms": decision.latency_ms,
        "usage": dict(usage),
        "request_id": decision.request_id,
    }
