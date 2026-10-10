"""Configurable final-answer adapters for EXPRESS-Bench."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .evidence_adapters import request_with_questions
from .intern_decision_engine import InternDecisionError


@dataclass
class AnswerResult:
    text: str
    status: str
    error_type: str | None = None
    raw: dict[str, Any] | None = None


class AnswerBackend:
    """Use Qwen by default, or bounded local Intern-Decision choices.

    Intern-Decision is a structured choice scorer, so open-ended benchmark
    answers require a static configured vocabulary. No vocabulary is inferred
    from the gold answer. Missing/invalid vocabularies intentionally fall back
    to the configured Qwen callback.
    """

    def __init__(
        self,
        *,
        mode: str = "qwen_vl",
        engine: Any = None,
        answer_options: Mapping[str, Any] | None = None,
        qwen_fn: Callable[..., str] | None = None,
    ) -> None:
        if mode not in {"qwen_vl", "intern_decision"}:
            raise ValueError("answer mode must be qwen_vl or intern_decision")
        self.mode = mode
        self.engine = engine
        self.answer_options = dict(answer_options or {})
        self.qwen_fn = qwen_fn

    def _qwen(self, question: str, question_type: str, state: Mapping[str, Any], error: Exception | None = None) -> AnswerResult:
        if self.qwen_fn is None:
            return AnswerResult("", "error", error_type=error.__class__.__name__ if error else "QwenUnavailable")
        try:
            text = self.qwen_fn(question, question_type, state)
            if not isinstance(text, str):
                raise TypeError("Qwen answer callback must return a string")
            return AnswerResult(text.strip(), "fallback_qwen" if error else "ok")
        except Exception as exc:
            return AnswerResult("", "error", error_type=exc.__class__.__name__)

    def _schema(self, question_type: str) -> tuple[str, dict[str, str]] | None:
        spec = self.answer_options.get(question_type, self.answer_options.get("default"))
        if spec is None:
            return None
        if isinstance(spec, Mapping) and "type" in spec:
            kind = str(spec.get("type"))
            criteria = spec.get("criteria")
            if kind == "noul":
                return kind, {"no": "The answer is no.", "yes": "The answer is yes."}
            if kind == "choice" and isinstance(criteria, Mapping):
                return kind, {str(k): str(v) for k, v in criteria.items()}
            return None
        if isinstance(spec, Mapping):
            criteria = {str(k): str(v) for k, v in spec.items()}
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            criteria = {str(value): str(value) for value in spec}
        else:
            return None
        if not criteria:
            return None
        return "choice", criteria

    def answer(self, question: str, question_type: str, state: Mapping[str, Any]) -> AnswerResult:
        if self.mode == "qwen_vl":
            return self._qwen(question, question_type, state)
        schema = self._schema(question_type)
        if schema is None or self.engine is None:
            return self._qwen(question, question_type, state, RuntimeError("no_static_answer_options"))
        kind, criteria = schema
        instructions = "Choose the answer that best answers the question from the configured options."
        request = request_with_questions(
            state,
            {"answer": {"type": kind, "instructions": instructions, "criteria": criteria}},
        )
        try:
            response = self.engine.predict(request)
            answers = response.get("answers") if isinstance(response, Mapping) else None
            answer = answers.get("answer") if isinstance(answers, Mapping) else None
            if not isinstance(answer, Mapping) or answer.get("type") != kind:
                raise ValueError("invalid Intern-Decision answer schema")
            if kind == "noul":
                value = float(answer.get("noul"))
                if not 0.0 <= value <= 1.0:
                    raise ValueError("invalid Intern-Decision noul probability")
                text = "yes" if value >= 0.5 else "no"
            else:
                text = answer.get("choice", answer.get("decision"))
                if not isinstance(text, str) or text not in criteria:
                    raise ValueError("Intern-Decision selected an unknown answer option")
            return AnswerResult(text, "ok", raw=dict(response))
        except Exception as exc:
            return self._qwen(question, question_type, state, exc)
