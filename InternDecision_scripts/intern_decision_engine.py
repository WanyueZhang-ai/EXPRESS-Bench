"""Lazy, local-only adapter for the Intern-Decision checkpoint.

The checkpoint ships its own ``inference.py``.  This adapter loads that module
relative to the configured checkpoint rather than relying on the current
working directory or a modified PYTHONPATH.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Type
from src.efficiency_models import model_call


DEFAULT_CHECKPOINT = "/ssd/wyzhang/all_models/Intern-Decision-4B"
DEFAULT_TEMPERATURE = 1.99241824


class InternDecisionError(RuntimeError):
    """Base class for local decision model errors."""


class InternDecisionLoadError(InternDecisionError):
    """The checkpoint or its inference implementation cannot be loaded."""


class InternDecisionInferenceError(InternDecisionError):
    """The loaded model rejected a request or failed during inference."""


def _repair_known_inference_syntax(source: str) -> str:
    replacements = {
        "f'{row.get('id', '<missing id>')}: questions must be a non-empty object'":
            "f\"{row.get('id', '<missing id>')}: questions must be a non-empty object\"",
        "f'{row.get('id', '<missing id>')}: question {field!r} is not an object'":
            "f\"{row.get('id', '<missing id>')}: question {field!r} is not an object\"",
        "f'{row.get('id', '<missing id>')}: question {field!r} has no options'":
            "f\"{row.get('id', '<missing id>')}: question {field!r} has no options\"",
        "f'{field_name}: {question.get('instructions', '')}'":
            "f\"{field_name}: {question.get('instructions', '')}\"",
    }
    for broken, fixed in replacements.items():
        source = source.replace(broken, fixed)
    return source


class InternDecisionEngine:
    """Load one local ``DecisionEngine`` and reuse it for all decisions.

    ``engine_cls`` is an injection point for offline tests.  Real runs resolve
    ``inference.py`` next to ``checkpoint`` and instantiate its
    ``DecisionEngine`` class with local-only settings.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        temperature: float = DEFAULT_TEMPERATURE,
        media_root: str | Path = ".",
        max_length: int = 8192,
        attn_implementation: str = "sdpa",
        inference_module: str | Path | None = None,
        engine_cls: Optional[Type[Any]] = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.inference_path = Path(inference_module).expanduser().resolve() if inference_module else self.checkpoint / "inference.py"
        if not self.checkpoint.is_dir():
            raise InternDecisionLoadError(f"Intern-Decision checkpoint directory does not exist: {self.checkpoint}")
        if not self.inference_path.is_file():
            raise InternDecisionLoadError(f"Intern-Decision inference.py does not exist: {self.inference_path}")
        if not (self.checkpoint / "config.json").is_file():
            raise InternDecisionLoadError(f"Intern-Decision checkpoint is missing config.json: {self.checkpoint}")
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if not isinstance(dtype, str) or dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")

        self.model_kwargs = {
            "checkpoint": str(self.checkpoint),
            "temperature": float(temperature),
            "backend": "hf",
            "media_root": str(Path(media_root).expanduser().resolve()),
            "max_length": max_length,
            "device": str(device),
            "dtype": dtype,
            "attn_implementation": str(attn_implementation),
        }
        cls = engine_cls or self._resolve_engine_class()
        try:
            self._engine = cls(**self.model_kwargs)
        except Exception as exc:  # pragma: no cover - real HF failures are environment-specific
            raise InternDecisionLoadError(
                f"Failed to load Intern-Decision from {self.checkpoint}: {exc.__class__.__name__}: {exc}"
            ) from exc

    def _resolve_engine_class(self) -> Type[Any]:
        module_name = f"express_bench_intern_decision_{abs(hash(str(self.inference_path)))}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, self.inference_path)
            if spec is None or spec.loader is None:
                raise ImportError("could not create a module spec")
            module = importlib.util.module_from_spec(spec)
            # ``exec_module`` does not register modules created from a file
            # spec.  Upstream inference.py uses ``@dataclass`` with postponed
            # annotations, whose Python 3.12 implementation looks the module
            # up in ``sys.modules`` while the class is being defined.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except SyntaxError:
                # A downloaded copy observed in the wild contains four nested
                # single-quote f-strings in compile_row. Repair only those
                # source literals in memory and never modify the checkpoint.
                source = self.inference_path.read_text(encoding="utf-8")
                source = _repair_known_inference_syntax(source)
                code = compile(source, str(self.inference_path), "exec")
                exec(code, module.__dict__)
            cls = getattr(module, "DecisionEngine")
        except Exception as exc:
            hint = ""
            if isinstance(exc, ImportError) and ("Qwen3_5" in str(exc) or "transformers" in str(exc)):
                hint = (
                    f" Install the checkpoint requirements with the same Python environment: "
                    f"pip install -r {self.checkpoint / 'requirements.txt'}"
                )
            raise InternDecisionLoadError(
                f"Failed to import local Intern-Decision inference module {self.inference_path}: "
                f"{exc.__class__.__name__}: {exc}.{hint}"
            ) from exc
        if not isinstance(cls, type):
            raise InternDecisionLoadError("local inference.py does not expose a DecisionEngine class")
        return cls

    @property
    def model(self) -> Any:
        return self._engine

    def predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run one immutable request through the local model."""
        if not isinstance(request, Mapping):
            raise TypeError("Intern-Decision request must be a mapping")
        images = request.get("images")
        images = images if isinstance(images, (list, tuple)) else []
        questions = request.get("questions")
        questions = questions if isinstance(questions, Mapping) else {}
        with model_call("intern_decision", "final_answer" if set(questions) == {"answer"} else "decision",
                        input_modality="text_image" if images else "text", output_kind="decision",
                        device=self.model_kwargs["device"], image_count=len(images),
                        checkpoint=str(self.checkpoint), candidate_count=sum(len(q["criteria"]) for q in questions.values() if isinstance(q, Mapping) and isinstance(q.get("criteria"), Mapping))) as call:
            request_copy = copy.deepcopy(dict(request))
            try:
                with call.observe(self._engine):
                    response = self._engine.predict(request_copy)
            except InternDecisionError:
                raise
            except Exception as exc:
                raise InternDecisionInferenceError(
                    f"Intern-Decision inference failed: {exc.__class__.__name__}: {exc}"
                ) from exc
            if not isinstance(response, Mapping):
                raise InternDecisionInferenceError("Intern-Decision response must be a mapping")
            call.set_response(response)
            return copy.deepcopy(dict(response))
