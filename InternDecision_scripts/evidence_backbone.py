"""One-forward structured Verifier/Value/Cost scoring for Intern-Decision."""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


_VALUE_BINS = {
    "0": "No expected new question-relevant visual evidence.",
    "0.25": "Little expected new question-relevant visual evidence.",
    "0.5": "Moderate expected new question-relevant visual evidence.",
    "0.75": "Substantial expected new question-relevant visual evidence.",
    "1": "Very strong expected new question-relevant visual evidence.",
}
_COST_BINS = {
    "0": "Negligible travel, time, and revisit cost.",
    "0.25": "Low travel, time, and revisit cost.",
    "0.5": "Moderate travel, time, and revisit cost.",
    "0.75": "High travel, time, and revisit cost.",
    "1": "Very high travel, time, and revisit cost.",
}

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _probability(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True)
class BackboneScores:
    verifier: float
    value: dict[str, float]
    cost: dict[str, float]
    decision_fields: int
    model_calls: int = 1
    latency_ms: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)


class EvidenceAwareDecisionBackbone:
    """One shared request for all score fields; never repeats image/context."""

    def __init__(self, engine: Any, *, max_candidates: int = 7) -> None:
        if not 1 <= int(max_candidates) <= 7:
            raise ValueError("max_candidates must be 1..7: checkpoint supports at most 16 fields")
        self.engine = engine
        self.max_candidates = int(max_candidates)
        self.last_predict_called = False
        self.last_decision_fields = 0

    def _questions(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        questions: dict[str, dict[str, Any]] = {
            "verifier": {
                "type": "noul",
                "instructions": "Is the visual evidence already observed sufficient to answer the task question accurately?",
                "criteria": {
                    "yes": "The answer is grounded in a currently observed or remembered RGB view.",
                    "no": "More observation is needed to ground the answer.",
                },
            }
        }
        for identifier in ids:
            questions[f"value_{identifier}"] = {
                "type": "score",
                "instructions": f"Rate expected new question-relevant visual evidence from candidate {identifier}, from 0 to 1.",
                "criteria": dict(_VALUE_BINS),
            }
            questions[f"cost_{identifier}"] = {
                "type": "score",
                "instructions": f"Rate travel/time/revisit cost of candidate {identifier}, from 0 to 1.",
                "criteria": dict(_COST_BINS),
            }
        return questions

    def score(self, state: Mapping[str, Any]) -> BackboneScores:
        self.last_predict_called = False
        self.last_decision_fields = 0
        if not isinstance(state, Mapping):
            raise ValueError("state must be a mapping")
        candidates = state.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) > self.max_candidates:
            raise ValueError("candidate count exceeds one-forward limit")
        ids = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate must be a mapping")
            identifier = candidate.get("id")
            if not isinstance(identifier, str) or not _ID_PATTERN.fullmatch(identifier):
                raise ValueError("candidate id is invalid")
            ids.append(identifier)
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids must be unique")
        request_state = copy.deepcopy(dict(state))
        images = request_state.pop("images", [])
        if not isinstance(images, list) or len(images) > 8:
            raise ValueError("images must be a list with at most 8 paths")
        request = {"state": request_state, "questions": self._questions(ids)}
        if images:
            request["images"] = copy.deepcopy(images)
        self.last_decision_fields = len(request["questions"])
        self.last_predict_called = True
        response = self.engine.predict(request)
        if not isinstance(response, Mapping) or not isinstance(response.get("answers"), Mapping):
            raise ValueError("model response is missing answers")
        answers = response["answers"]
        verifier_answer = answers.get("verifier")
        if not isinstance(verifier_answer, Mapping) or verifier_answer.get("type") != "noul":
            raise ValueError("model response is missing verifier")
        verifier = _probability(verifier_answer.get("noul"), "verifier")
        value: dict[str, float] = {}
        cost: dict[str, float] = {}
        raw: dict[str, dict[str, float]] = {}
        for identifier in ids:
            for kind, destination in (("value", value), ("cost", cost)):
                field = f"{kind}_{identifier}"
                answer = answers.get(field)
                if not isinstance(answer, Mapping) or answer.get("type") != "score":
                    raise ValueError(f"model response is missing {field}")
                destination[identifier] = _probability(answer.get("score"), field)
                probabilities = answer.get("probabilities", {})
                if isinstance(probabilities, Mapping):
                    raw[field] = {
                        str(key): _probability(number, f"{field}.{key}")
                        for key, number in probabilities.items()
                    }
        timing = response.get("timing", {})
        latency = timing.get("inference_ms") if isinstance(timing, Mapping) else None
        if latency is not None:
            latency = float(latency)
            if not math.isfinite(latency) or latency < 0:
                latency = None
        usage = response.get("usage", {})
        return BackboneScores(
            verifier=verifier, value=value, cost=cost,
            decision_fields=len(request["questions"]), latency_ms=latency,
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            raw_probabilities=raw,
        )
