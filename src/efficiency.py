"""Opt-in wall-clock accounting for calls, exploration cycles and episodes.

This module deliberately has no Habitat, NumPy, Torch or Transformers dependency.
Nested spans use exclusive wall time for cost accounting; CUDA timeline timings
are separate measurements and must never be added to wall time.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from typing import Any, Mapping


SCHEMA_VERSION = "express-efficiency-v1"
TRACE_FILES = {
    "call": "efficiency_calls.jsonl",
    "step": "efficiency_steps.jsonl",
    "episode": "efficiency_episodes.jsonl",
    "run": "efficiency_runs.jsonl",
}
REPORT_FILES = ("efficiency_summary.json", "efficiency_summary.csv", "efficiency_report.html")
_ACTIVE = contextvars.ContextVar("express_efficiency", default=None)
_SPAN = contextvars.ContextVar("express_efficiency_span", default=None)


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _cuda(device):
    torch = sys.modules.get("torch")
    if torch is None or device is None or not str(device).startswith("cuda"):
        return None
    return torch.cuda if torch.cuda.is_available() else None


class NullScope:
    enabled = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **values):
        pass

    def elapsed_ms(self):
        return 0.0


class NullProfiler:
    enabled = False
    mode = "off"

    def span(self, *args, **kwargs):
        return NullScope()

    step = episode = span
    begin_step = begin_episode = span

    def end_step(self, *exc):
        pass

    end_episode = end_step

    def measure(self, name, function, *args, category=None, **kwargs):
        return function(*args, **kwargs)


_NULL = NullProfiler()


def current_profiler():
    return _ACTIVE.get() or _NULL


class Scope:
    enabled = True

    def __init__(self, profiler, name, *, kind="span", category=None, device=None, **info):
        self.profiler, self.name, self.kind = profiler, name, kind
        self.category, self.device = category, device
        self.info = dict(info)
        self.stages = {}
        self.categories = defaultdict(float)
        self.child_ms = 0.0
        self.call_ids = []
        self.token_counts = defaultdict(int)
        self.token_known_calls = defaultdict(int)
        self.parent = None
        self.gpu_events = None

    def update(self, **values):
        self.info.update(values)

    def elapsed_ms(self):
        return max(0.0, (self.profiler.clock() - self.started) * 1000.0)

    def __enter__(self):
        self.parent = _SPAN.get()
        self.category = self.category or (self.parent.category if self.parent else "other")
        inherited = {}
        if self.parent:
            inherited = {k: self.parent.info[k] for k in ("episode_id", "attempt_id", "step", "question_type") if k in self.parent.info}
        self.info = {**inherited, **self.info}
        self.id = uuid.uuid4().hex
        cuda = _cuda(self.device) if self.profiler.mode == "detailed" else None
        if cuda:
            try:
                cuda.synchronize(self.device)
                with cuda.device(self.device):
                    self.gpu_events = (cuda.Event(enable_timing=True), cuda.Event(enable_timing=True))
                    self.gpu_events[0].record()
            except Exception as error:
                self.info["gpu_timing_error"] = type(error).__name__
                self.gpu_events = None
        self.started = self.profiler.clock()
        self.started_unix = time.time()
        if self.kind == "step":
            self._memory_start()
        self._context_token = _SPAN.set(self)
        return self

    def _memory_start(self):
        device = self.info.get("memory_device")
        cuda = _cuda(device)
        if cuda:
            try:
                self.info["memory"] = {"device": str(device), "gpu_name": cuda.get_device_name(device),
                                       "baseline_allocated_bytes": cuda.memory_allocated(device),
                                       "baseline_reserved_bytes": cuda.memory_reserved(device),
                                       "scope": "process_allocator_including_all_resident_models"}
                cuda.reset_peak_memory_stats(device)
            except Exception as error:
                self.info["memory_error"] = type(error).__name__

    def _finish(self, exc_type):
        memory = self.info.get("memory")
        if memory:
            try:
                cuda = _cuda(memory["device"])
                memory["peak_allocated_bytes"] = cuda.max_memory_allocated(memory["device"])
                memory["peak_reserved_bytes"] = cuda.max_memory_reserved(memory["device"])
                memory["peak_increment_bytes"] = max(0, memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"])
            except Exception as error:
                self.info["memory_error"] = type(error).__name__
        gpu_ms = None
        if self.gpu_events:
            try:
                cuda = _cuda(self.device)
                with cuda.device(self.device):
                    self.gpu_events[1].record()
                    self.gpu_events[1].synchronize()
                    gpu_ms = self.gpu_events[0].elapsed_time(self.gpu_events[1])
            except Exception as error:
                self.info["gpu_timing_error"] = type(error).__name__
        self.ended = self.profiler.clock()
        self.wall_ms = max(0.0, (self.ended - self.started) * 1000.0)
        exclusive_ms = max(0.0, self.wall_ms - self.child_ms)
        self.categories[self.category] += exclusive_ms
        self.info.setdefault("status", "error" if exc_type else "ok")
        if exc_type:
            self.info["status"] = "error"
            self.info["error_type"] = exc_type.__name__
        ancestor = self.parent
        if ancestor:
            ancestor.child_ms += self.wall_ms
        while ancestor:
            stage = ancestor.stages.setdefault(self.name, {"wall_ms": 0.0, "exclusive_wall_ms": 0.0, "gpu_timeline_ms": None, "count": 0})
            stage["wall_ms"] += self.wall_ms
            stage["exclusive_wall_ms"] += exclusive_ms
            stage["count"] += 1
            if gpu_ms is not None:
                stage["gpu_timeline_ms"] = (stage["gpu_timeline_ms"] or 0.0) + gpu_ms
            ancestor.categories[self.category] += exclusive_ms
            ancestor = ancestor.parent
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.profiler.run_id,
            "worker_id": self.profiler.worker_id,
            "condition": self.profiler.condition,
            "record_id": self.id,
            "kind": self.kind,
            "name": self.name,
            "mode": self.profiler.mode,
            "started_at_unix": self.started_unix,
            "finished_at_unix": time.time(),
            "wall_ms": self.wall_ms,
            "exclusive_wall_ms": exclusive_ms,
            "gpu_timeline_ms": gpu_ms,
            "stages": self.stages,
            "category_wall_ms": dict(self.categories),
            **self.info,
        }

    def __exit__(self, exc_type, exc, traceback):
        try:
            record = self._finish(exc_type)
            if self.kind in {"step", "episode"}:
                record.update({
                    "model_call_count": len(self.call_ids),
                    "model_call_ids": self.call_ids,
                    # A total is null unless every call supplies that metric.
                    "tokens": {k: self.token_counts[k] if self.token_known_calls[k] == len(self.call_ids) else None for k in TOKEN_FIELDS},
                    "known_token_subtotals": dict(self.token_counts),
                    "token_coverage_calls": {k: self.token_known_calls[k] for k in TOKEN_FIELDS},
                })
                if self.kind == "step":
                    record["simulator_transition_count"] = self.stages.get("simulator.action", {}).get("count", 0)
                    record["observation_count"] = self.stages.get("simulator.observe", {}).get("count", 0)
            if self.kind in TRACE_FILES:
                self.profiler.write(self.kind, record)
        finally:
            _SPAN.reset(self._context_token)
        return False


TOKEN_FIELDS = ("text_input_tokens", "image_input_tokens", "effective_input_tokens", "generated_tokens")


class EfficiencyProfiler:
    enabled = True

    def __init__(self, output_dir, config=None, *, append=False, clock=time.perf_counter, metadata=None):
        config = config or {}
        self.output_dir = Path(output_dir)
        self.mode = str(config.get("mode", "light"))
        if self.mode not in {"light", "detailed"}:
            raise ValueError("efficiency.mode must be light or detailed")
        self.warmup_calls = int(config.get("warmup_calls", 1))
        if self.warmup_calls < 0:
            raise ValueError("efficiency.warmup_calls must be nonnegative")
        self.report_enabled = bool(config.get("report", True))
        self.run_id = uuid.uuid4().hex
        self.worker_id = str(config.get("worker_id", self.output_dir.name))
        self.clock = clock
        self.metadata = metadata or {}
        intern = self.metadata.get("intern", {})
        jev = self.metadata.get("jev", {})
        self.condition = ("intern:" + str(intern.get("evidence_mode")) + ":" + str(intern.get("action_mode")) if intern.get("enabled")
                          else "jev:" + str(jev.get("mode")) if jev.get("enabled") else "baseline")
        self.model_calls = defaultdict(int)
        self.write_ms = 0.0
        self.io_errors = []
        self._episode = self._step = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for filename in TRACE_FILES.values():
            path = self.output_dir / filename
            if not append:
                path.write_text("", encoding="utf-8")
            elif path.exists():
                self._repair_tail(path)

    @staticmethod
    def _repair_tail(path):
        """Keep valid records when a killed worker left half a final JSON line."""
        with path.open("rb+") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            if not end:
                return
            handle.seek(end - 1)
            if handle.read(1) == b"\n":
                return
            position = end
            tail = b""
            while position > 0:
                count = min(position, 8192)
                position -= count
                handle.seek(position)
                tail = handle.read(count) + tail
                if b"\n" in tail:
                    tail = tail.rsplit(b"\n", 1)[1]
                    break
            try:
                json.loads(tail)
            except (ValueError, UnicodeDecodeError):
                # Preserve the damaged bytes for inspection before truncation.
                recovery = path.with_name(path.name + ".truncated-" + uuid.uuid4().hex[:8])
                recovery.write_bytes(tail)
                handle.seek(end - len(tail))
                handle.truncate()
                logging.warning("Recovered truncated efficiency trace %s; tail kept at %s", path, recovery)
            else:
                handle.seek(0, 2)
                handle.write(b"\n")

    def write(self, kind, record):
        started = self.clock()
        try:
            with (self.output_dir / TRACE_FILES[kind]).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        except OSError as error:
            self.io_errors.append(type(error).__name__)
            logging.warning("Efficiency trace write failed: %s", error)
        self.write_ms += max(0.0, (self.clock() - started) * 1000.0)

    def span(self, name, **kwargs):
        return Scope(self, name, **kwargs)

    def episode(self, episode_id, **info):
        return self.span("episode", kind="episode", category="other", episode_id=episode_id, attempt_id=uuid.uuid4().hex, **info)

    def step(self, step, **info):
        return self.span("step", kind="step", category="other", step=step, **info)

    def begin_episode(self, episode_id, **info):
        self.end_episode()
        self._episode = self.episode(episode_id, **info)
        return self._episode.__enter__()

    def begin_step(self, step, **info):
        self.end_step()
        self._step = self.step(step, **info)
        return self._step.__enter__()

    def end_step(self, exc_type=None, exc=None, traceback=None):
        scope, self._step = self._step, None
        if scope:
            scope.__exit__(exc_type, exc, traceback)

    def end_episode(self, exc_type=None, exc=None, traceback=None):
        self.end_step(exc_type, exc, traceback)
        scope, self._episode = self._episode, None
        if scope:
            scope.__exit__(exc_type, exc, traceback)

    def measure(self, name, function, *args, category=None, **kwargs):
        with self.span(name, category=category):
            return function(*args, **kwargs)

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            self.end_episode(*sys.exc_info())
            _ACTIVE.reset(token)

    def finish(self):
        if self.report_enabled:
            from src.efficiency_report import build_report
            build_report([self.output_dir], self.output_dir)


def _runtime_metadata(cfg):
    versions = {}
    for package in ("torch", "transformers", "habitat-sim", "numpy", "prismatic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        revision = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    # Explicit allowlist: never serialize API keys, hf_token or prompt contents.
    intern = cfg.get("intern_decision", {}) or {}
    vlm = cfg.get("vlm", {}) or {}
    jev = cfg.get("jev", {}) or {}
    root = Path(__file__).resolve().parents[1]
    prompt_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((root / "prompt").glob("*.txt"))}
    checkpoint = Path(str(intern.get("checkpoint", "")))
    inference_hash = None
    if (checkpoint / "inference.py").is_file():
        inference_hash = hashlib.sha256((checkpoint / "inference.py").read_bytes()).hexdigest()
    return {
        "git_revision": revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": cfg.get("seed"),
        "prompt_sha256": prompt_hashes,
        "intern_inference_sha256": inference_hash,
        "qwen_model_path": getattr(sys.modules.get("gpt"), "QWEN3_VL_MODEL_PATH", None),
        "benchmark_config": {k: cfg.get(k) for k in ("dataset_path", "camera_height", "camera_tilt_deg", "hfov", "tsdf_grid_size", "max_step_room_size_ratio", "black_pixel_ratio", "min_random_init_steps", "planner", "visual_prompt")},
        "sensor_size": [cfg.get("img_width"), cfg.get("img_height")],
        "save_obs": cfg.get("save_obs"),
        "intern": {k: intern.get(k) for k in ("enabled", "checkpoint", "device", "dtype", "evidence_mode", "action_mode", "answer_backend", "temperature", "max_length", "attn_implementation")},
        "vlm": {k: vlm.get(k) for k in ("model_id", "device")},
        "jev": {k: jev.get(k) for k in ("enabled", "model", "mode", "max_retries", "timeout_seconds")},
    }


def profiled_run(function):
    """Install one worker-local collector; always finalize on return/error."""
    @functools.wraps(function)
    def wrapper(cfg, *args, **kwargs):
        config = cfg.get("efficiency", {}) or {}
        if not bool(config.get("enabled", False)):
            return function(cfg, *args, **kwargs)
        profiler = EfficiencyProfiler(cfg.output_dir if hasattr(cfg, "output_dir") else cfg["output_dir"], config, append=bool(kwargs.get("append_trace", False)), metadata=_runtime_metadata(cfg))
        token = _ACTIVE.set(profiler)
        try:
            with profiler.span("run", kind="run", category="other", metadata=profiler.metadata, warmup_calls=profiler.warmup_calls) as run:
                try:
                    return function(cfg, *args, **kwargs)
                finally:
                    profiler.end_episode(*sys.exc_info())
                    run.update(profiling_write_ms=profiler.write_ms, profiling_io_errors=profiler.io_errors)
        finally:
            _ACTIVE.reset(token)
            try:
                profiler.finish()
            except Exception:
                # Keep the original benchmark exception and raw data available.
                logging.exception("Efficiency report failed; raw traces are preserved")
    return wrapper


def apply_efficiency_cli(cfg, *, enabled=False, mode=None):
    if enabled or mode is not None:
        if cfg.get("efficiency") is None:
            cfg["efficiency"] = {}
        cfg["efficiency"]["enabled"] = True
        if mode is not None:
            cfg["efficiency"]["mode"] = mode


def profiled_operation(name, category):
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with current_profiler().span(name, category=category):
                return function(*args, **kwargs)
        return wrapper
    return decorate
