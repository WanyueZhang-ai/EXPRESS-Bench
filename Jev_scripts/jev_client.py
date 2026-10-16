"""Small Python 3.9 compatible client for the TypeSafe System One API.

The repository intentionally does not require the TypeSafe SDK.  Jev's HTTP
contract is simple enough to keep the experiment self-contained and auditable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional
from src.efficiency import current_profiler
from src.efficiency_models import model_call


DEFAULT_BASE_URL = "https://api.typesafe.ai/v1/systemone"


class JevError(RuntimeError):
    """Base class for Jev client failures."""


JevApiError = JevError


class JevConfigurationError(JevError):
    """The client cannot be configured safely."""


class JevTransportError(JevError):
    """A network or connection failure occurred."""


class JevHTTPError(JevError):
    """The API returned a non-success HTTP status."""

    def __init__(self, status: int, body: str = "") -> None:
        self.status = int(status)
        self.body = body
        super().__init__("TypeSafe API returned HTTP {}".format(self.status))


class JevSchemaError(JevError):
    """The API response did not match the System One response contract."""


@dataclass
class JevResponse:
    model: str
    answers: Dict[str, Dict[str, Any]]
    usage: Dict[str, Any]
    latency_ms: float
    request_id: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def answer(self, question_id: str) -> Optional[Dict[str, Any]]:
        return self.answers.get(question_id)


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


class JevClient:
    """Call ``/v1/systemone`` with bounded, reproducible retry behavior.

    ``transport`` is injectable so request construction and response parsing can
    be tested without making a network call.  It receives URL, headers, JSON
    payload, and timeout seconds and returns a decoded JSON mapping.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "jev-latest",
        timeout: float = 20.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        transport: Optional[Transport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key or not self.api_key.strip():
            raise JevConfigurationError(
                "TYPESAFE_API_KEY is required; pass api_key explicitly or set the environment variable"
            )
        if timeout <= 0:
            raise JevConfigurationError("timeout must be positive")
        if max_retries < 0:
            raise JevConfigurationError("max_retries must be non-negative")
        self.base_url = base_url
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self._transport = transport or self._urllib_transport
        self._sleep = sleep

    @staticmethod
    def _urllib_transport(
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Mapping[str, Any]:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise JevHTTPError(exc.code, body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JevTransportError(str(exc))
        try:
            decoded = json.loads(body)
        except (TypeError, ValueError) as exc:
            raise JevSchemaError("TypeSafe API returned invalid JSON: {}".format(exc))
        if not isinstance(decoded, dict):
            raise JevSchemaError("TypeSafe API response must be a JSON object")
        return decoded

    @staticmethod
    def _retryable(error: Exception) -> bool:
        if isinstance(error, JevTransportError):
            return True
        return isinstance(error, JevHTTPError) and error.status in {
            408,
            425,
            429,
            500,
            502,
            503,
            504,
        }

    def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> JevResponse:
        if state is None:
            raise JevConfigurationError("state cannot be None")
        if not isinstance(questions, Mapping) or not questions:
            raise JevConfigurationError("questions must be a non-empty mapping")
        payload = {"state": state, "model": self.model, "questions": dict(questions)}
        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        efficiency = current_profiler()
        with model_call("jev", "decision", input_modality="text", output_kind="decision",
                        backend_model=self.model, decision_fields=len(questions)) as call:
            started = time.perf_counter()
            last_error: Optional[Exception] = None
            for attempt in range(self.max_retries + 1):
                call.update(attempts=attempt + 1, retry_count=attempt)
                try:
                    raw = efficiency.measure("api.transport", self._transport, self.base_url, headers, payload, self.timeout)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    response = self.parse_response(raw, latency_ms=elapsed_ms)
                    call.set_response(raw, external=True)
                    return response
                except (JevTransportError, JevHTTPError, JevSchemaError) as exc:
                    last_error = exc
                    if attempt >= self.max_retries or not self._retryable(exc):
                        raise
                    efficiency.measure("api.retry_wait", self._sleep, self.backoff_seconds * (2 ** attempt))
            raise last_error or JevError("Jev request failed")

    @classmethod
    def parse_response(
        cls,
        raw: Mapping[str, Any],
        *,
        latency_ms: float = 0.0,
    ) -> JevResponse:
        if not isinstance(raw, Mapping):
            raise JevSchemaError("response must be a JSON object")
        answers = raw.get("answers")
        if not isinstance(answers, Mapping):
            raise JevSchemaError("response.answers must be an object")
        normalized: Dict[str, Dict[str, Any]] = {}
        for question_id, answer in answers.items():
            if not isinstance(question_id, str) or not isinstance(answer, Mapping):
                raise JevSchemaError("each answer must be an object keyed by question id")
            answer_type = answer.get("type")
            if answer_type not in {"noul", "choice", "score"}:
                raise JevSchemaError("unknown answer type for {}".format(question_id))
            item = dict(answer)
            if answer_type == "noul":
                value = item.get("noul")
                if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                    raise JevSchemaError("noul answer {} must be in [0, 1]".format(question_id))
                item["noul"] = float(value)
            elif answer_type == "choice":
                if not isinstance(item.get("choice"), str):
                    raise JevSchemaError("choice answer {} is missing choice".format(question_id))
                if "probabilities" in item:
                    probabilities = item["probabilities"]
                    if not isinstance(probabilities, Mapping):
                        raise JevSchemaError("choice probabilities must be an object")
                    try:
                        item["probabilities"] = {
                            str(key): float(value) for key, value in probabilities.items()
                        }
                    except (TypeError, ValueError) as exc:
                        raise JevSchemaError(
                            "choice probabilities for {} must be numeric: {}".format(
                                question_id, exc
                            )
                        )
                if "confidence" in item:
                    try:
                        item["confidence"] = float(item["confidence"])
                    except (TypeError, ValueError) as exc:
                        raise JevSchemaError(
                            "choice confidence for {} must be numeric: {}".format(
                                question_id, exc
                            )
                        )
            else:
                if not isinstance(item.get("score"), (int, float)):
                    raise JevSchemaError("score answer {} is missing score".format(question_id))
                item["score"] = float(item["score"])
            normalized[question_id] = item
        usage = raw.get("usage", {})
        if usage is None:
            usage = {}
        if not isinstance(usage, Mapping):
            raise JevSchemaError("response.usage must be an object")
        model = raw.get("model", "")
        if not isinstance(model, str):
            raise JevSchemaError("response.model must be a string")
        request_id = raw.get("request_id", raw.get("id"))
        if request_id is not None and not isinstance(request_id, str):
            request_id = str(request_id)
        return JevResponse(
            model=model,
            answers=normalized,
            usage=dict(usage),
            latency_ms=float(latency_ms),
            request_id=request_id,
            raw=dict(raw),
        )
