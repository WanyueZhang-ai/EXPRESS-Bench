"""Non-generative/generative model probes. Hooks never change model outputs."""
from __future__ import annotations

from contextlib import AbstractContextManager
from collections.abc import Mapping

from src.efficiency import Scope, TOKEN_FIELDS, current_profiler


class NullModelCall:
    enabled = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **values):
        pass

    def set_inputs(self, *args, **kwargs):
        pass

    def set_generated(self, *args, **kwargs):
        pass

    def set_response(self, *args, **kwargs):
        pass

    def observe(self, model):
        return self

    def streamer_kwargs(self):
        return {}


def model_call(model, purpose, *, input_modality, output_kind, device=None, **metadata):
    profiler = current_profiler()
    if not profiler.enabled:
        return NullModelCall()
    return ModelCall(profiler, model, purpose, input_modality=input_modality, output_kind=output_kind, device=device, **metadata)


def _tensor_rows(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    value = [item.detach().cpu().tolist() if hasattr(item, "detach") else item.tolist() if hasattr(item, "tolist") else item for item in value]
    if not value:
        return []
    return list(value) if isinstance(value[0], (list, tuple)) else [list(value)]


def _module_root(obj):
    """Find the real HF/Prismatic module behind a checkpoint-owned adapter."""
    queue, seen = [(obj, "")], set()
    while queue and len(seen) < 40:
        candidate, path = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if callable(getattr(candidate, "register_forward_hook", None)) and callable(getattr(candidate, "named_modules", None)):
            return candidate, path
        if path.count(".") < 4:
            for key in ("model", "_model", "backend", "_backend", "engine", "_engine", "vlm", "hf_model"):
                child = getattr(candidate, key, None)
                if child is not None and not isinstance(child, (str, int, float, bool)):
                    queue.append((child, (path + "." + key).lstrip(".")))
    return None, None


def _visual_ids(root):
    configs = [getattr(root, "config", None), getattr(getattr(root, "model", None), "config", None)]
    return {int(value) for config in configs for value in [getattr(config, "image_token_id", None)] if isinstance(value, int) and value >= 0}


def _feature_tokens(output):
    """Count projected feature rows, not raw patches or parallel tower outputs."""
    for key in ("pooler_output", "image_embeds", "last_hidden_state"):
        value = output.get(key) if isinstance(output, Mapping) else getattr(output, key, None)
        if value is not None:
            return _feature_tokens(value)
    shape = getattr(output, "shape", ())
    if len(shape) == 3:
        return int(shape[0]) * int(shape[1])
    if len(shape) == 2:
        return int(shape[0])
    # Some HF visual encoders return (merged_features, deepstack_features).
    # Auxiliary deepstack features do not add positions to the LLM sequence.
    if isinstance(output, tuple) and output:
        return _feature_tokens(output[0])
    return None


class _TokenObserver:
    """HF streamer protocol; records receipt of actual generated token IDs."""
    def __init__(self, call):
        self.call = call
        self.prompt = True

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        if self.call.first_token_at is None:
            self.call.first_token_at = self.call.profiler.clock()
        rows = _tensor_rows(value)
        if rows is not None:
            self.call.tokens["generated_tokens"] = (self.call.tokens["generated_tokens"] or 0) + sum(len(row) for row in rows)
            self.call.sources["generated_tokens"] = "streamed_new_output_ids_including_special_tokens"

    def end(self):
        self.call.generation_finished_at = self.call.profiler.clock()


class ModelCall(Scope):
    def __init__(self, profiler, model, purpose, *, input_modality, output_kind, device=None, **metadata):
        category = "decision_model" if output_kind == "decision" else "vlm"
        super().__init__(profiler, "model." + model, kind="call", category=category, device=device,
                         model=model, purpose=purpose, input_modality=input_modality, output_kind=output_kind,
                         **metadata)
        self.info["device"] = str(device) if device is not None else None
        if purpose == "eac_evaluation":
            self.category = "evaluation"
        self.tokens = {key: None for key in TOKEN_FIELDS}
        self.sources = {}
        self.has_image = input_modality == "text_image"
        if not self.has_image:
            self.tokens["image_input_tokens"] = 0
            self.sources["image_input_tokens"] = "text_only"
        if output_kind != "generation":
            self.tokens["generated_tokens"] = 0
            self.sources["generated_tokens"] = "non_generative_forward"
        self.image_ready_at = None
        self.first_token_at = None
        self.generation_finished_at = None
        self.forward_count = 0
        self.encoding_count = 0
        self.inputs_recorded = False
        self.layout = "inline"
        self.observations = []

    def __enter__(self):
        key = (self.info["model"], self.info["purpose"], self.info["input_modality"])
        count = self.profiler.model_calls[key]
        self.profiler.model_calls[key] += 1
        self.info.update(call_index=count, warmup=count < self.profiler.warmup_calls)
        return super().__enter__()

    def set_inputs(self, inputs, *, visual_token_ids=None, layout="inline", source="model_input_ids"):
        if self.inputs_recorded:
            return
        ids = inputs.get("input_ids") if hasattr(inputs, "get") else getattr(inputs, "input_ids", None)
        rows = _tensor_rows(ids)
        if rows is None:
            return
        attention = inputs.get("attention_mask") if hasattr(inputs, "get") else getattr(inputs, "attention_mask", None)
        masks = _tensor_rows(attention)
        active = [token for i, row in enumerate(rows) for j, token in enumerate(row) if masks is None or masks[i][j]]
        self.layout = layout
        self.inputs_recorded = True
        if layout == "inserted":
            self.tokens["text_input_tokens"] = len(active)
            self.sources["text_input_tokens"] = source
            self._complete_inserted_tokens()
        elif not self.has_image:
            self.tokens.update(text_input_tokens=len(active), effective_input_tokens=len(active))
            self.sources.update(text_input_tokens=source, effective_input_tokens=source)
        elif visual_token_ids:
            image_count = sum(token in visual_token_ids for token in active)
            self.tokens.update(text_input_tokens=len(active) - image_count, image_input_tokens=image_count, effective_input_tokens=len(active))
            self.sources.update({key: source + ":expanded_visual_positions" for key in ("text_input_tokens", "image_input_tokens", "effective_input_tokens")})
        else:
            # Do not label unknown multimodal placeholders as text or features.
            self.tokens["effective_input_tokens"] = len(active)
            self.sources["effective_input_tokens"] = source + ":modality_split_unavailable"

    def _complete_inserted_tokens(self):
        text, image = self.tokens["text_input_tokens"], self.tokens["image_input_tokens"]
        if text is not None and image is not None:
            self.tokens["effective_input_tokens"] = text + image
            self.sources["effective_input_tokens"] = "text_ids_plus_projected_feature_positions"

    def set_generated(self, generated_ids):
        rows = _tensor_rows(generated_ids)
        if rows is not None:
            self.tokens["generated_tokens"] = sum(len(row) for row in rows)
            self.sources["generated_tokens"] = "new_output_ids_including_special_tokens"
        self.generation_finished_at = self.profiler.clock()

    def set_response(self, response, *, external=False):
        if not isinstance(response, Mapping):
            return
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            # Retain the backend contract separately from observed token counts.
            self.info["reported_usage"] = dict(usage)
            if external:
                for field, aliases in {"effective_input_tokens": ("input_tokens", "prompt_tokens"), "generated_tokens": ("output_tokens", "completion_tokens")}.items():
                    for key in aliases:
                        value = usage.get(key)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            self.tokens[field] = value
                            self.sources[field] = "provider_reported:" + key
                            break
        if isinstance(response.get("answers"), Mapping):
            self.info["decision_fields"] = len(response["answers"])
        if isinstance(response.get("timing"), Mapping):
            self.info["reported_timing"] = dict(response["timing"])

    def observe(self, model):
        observation = ModelObservation(self, model)
        self.observations.append(observation)
        return observation

    def streamer_kwargs(self):
        if self.profiler.mode != "detailed":
            return {}
        self.info["token_streamer_enabled"] = True
        self.info["token_streamer_note"] = "HF streamer copies tokens to CPU; detailed timing includes this overhead"
        return {"streamer": _TokenObserver(self)}

    def _finish(self, exc_type):
        record = super()._finish(exc_type)
        record.update(tokens=self.tokens, token_sources=self.sources, model_forward_count=self.forward_count,
                      image_encoding_count=self.encoding_count)
        if not self.has_image:
            status = "not_applicable"
        elif self.profiler.mode != "detailed":
            status = "not_instrumented_in_light_mode"
        elif self.image_ready_at is None:
            status = "unavailable"
        else:
            status = "observed"
        record["image_encoding_status"] = status
        record["input_to_image_encoded_ms"] = None if self.image_ready_at is None else max(0.0, (self.image_ready_at - self.started) * 1000)
        record["image_encoded_to_output_ms"] = None if self.image_ready_at is None else max(0.0, (self.ended - self.image_ready_at) * 1000)
        if status == "unavailable":
            record["image_encoding_unavailable_reason"] = self.info.get("probe_unavailable_reason", "no_completed_visual_encoder_forward")
        record["input_to_first_token_ms"] = None if self.first_token_at is None else (self.first_token_at - self.started) * 1000
        record["ttft_status"] = "observed" if self.first_token_at is not None else ("not_applicable" if self.info["output_kind"] != "generation" else "not_observed")
        count = self.tokens["generated_tokens"]
        stage = self.stages.get("generation", {})
        duration = stage.get("wall_ms", 0)
        record["generation_tokens_per_second"] = count * 1000 / duration if count is not None and duration > 0 else None
        after_first = None
        if self.first_token_at is not None and self.generation_finished_at is not None and count is not None and count > 1:
            duration = self.generation_finished_at - self.first_token_at
            if duration > 0:
                after_first = (count - 1) / duration
        record["decode_tokens_per_second"] = after_first
        ancestor = self.parent
        while ancestor:
            ancestor.call_ids.append(self.id)
            for key, value in self.tokens.items():
                if value is not None:
                    ancestor.token_counts[key] += value
                    ancestor.token_known_calls[key] += 1
            ancestor = ancestor.parent
        return record

    def __exit__(self, exc_type, exc, traceback):
        for observation in reversed(self.observations):
            observation.close(exc_type, exc, traceback)
        return super().__exit__(exc_type, exc, traceback)


class ModelObservation(AbstractContextManager):
    def __init__(self, call, target):
        self.call, self.target = call, target
        self.handles = []
        self.pending = []
        self.method_restores = []

    def __enter__(self):
        try:
            root, root_path = _module_root(self.target)
            ids = _visual_ids(root) if root is not None else set()
        except Exception as error:
            # Checkpoint-owned wrappers may expose properties that raise.
            # An unavailable probe must not prevent the actual inference.
            self.call.update(probe_unavailable_reason="module_discovery_failed:" + type(error).__name__)
            return self
        if root is None:
            self.call.update(probe_unavailable_reason="checkpoint_adapter_exposes_no_torch_module")
            return self
        # Capture prepared inputs even in light mode. This only reads IDs.
        def inputs(module, args, kwargs):
            self.call.forward_count += 1
            try:
                values = dict(kwargs)
                if "input_ids" not in values and args:
                    values["input_ids"] = args[0]
                self.call.set_inputs(values, visual_token_ids=ids, layout=self.call.layout)
            except Exception as error:
                self.call.update(token_probe_error=type(error).__name__)
        try:
            self.handles.append(root.register_forward_pre_hook(inputs, with_kwargs=True))
            modules = dict(root.named_modules())
            def find(names):
                candidates = [(path, module) for path, module in modules.items() if path and path.rsplit(".", 1)[-1] in names]
                return min(candidates, key=lambda pair: (pair[0].count("."), len(pair[0]))) if candidates else (None, None)
            vision_path, vision = find({"visual", "vision_backbone", "vision_tower", "vision_model"})
            projector_path, projector = find({"projector", "multi_modal_projector", "mm_projector"})
            if vision_path and projector_path and projector_path.startswith(vision_path + "."):
                projector_path, projector = None, None
            language_path, language = find({"language_model", "llm_backbone"})
            if self.call.profiler.mode != "detailed":
                if projector is not None and self.call.layout == "inserted":
                    def feature_count(module, args, output):
                        count = _feature_tokens(output)
                        if count is not None:
                            self.call.tokens["image_input_tokens"] = count
                            self.call.sources["image_input_tokens"] = "projector_output_sequence_shape"
                            self.call._complete_inserted_tokens()
                    self.handles.append(projector.register_forward_hook(feature_count))
                return self
            # Current Intern HF adapters expose encode() on the backend. Wrap
            # only that instance for this call, preserving checkpoint files.
            owner = self.target
            for component in (root_path or "").split("."):
                if callable(getattr(owner, "encode", None)):
                    self._prepare_hook(owner)
                    break
                owner = getattr(owner, component, None)
                if owner is None:
                    break
            self.call.update(observed_module_paths={"root": root_path or "<root>", "vision": vision_path, "projector": projector_path, "language": language_path})
            if vision is None and self.call.has_image:
                self.call.update(probe_unavailable_reason="unsupported_visual_module_layout")
            self._hook(root, "model.forward")
            if vision is not None:
                self._hook(vision, "vision.encode", image_ready=projector is None)
            if projector is not None:
                self._hook(projector, "vision.project", image_ready=True, projected=True)
            if language is not None and language is not root:
                self._hook(language, "language.forward")
        except Exception as error:
            self.call.update(probe_unavailable_reason="hook_registration_failed:" + type(error).__name__)
            self.close(None, None, None)
        return self

    def _prepare_hook(self, owner):
        original = owner.encode
        own = "encode" in getattr(owner, "__dict__", {})
        def encode(*args, **kwargs):
            with self.call.profiler.span("input.prepare"):
                return original(*args, **kwargs)
        owner.encode = encode
        self.method_restores.append((owner, original, own))

    def _hook(self, module, name, *, image_ready=False, projected=False):
        def before(current, args):
            scope = self.call.profiler.span(name, device=self.call.device)
            scope.__enter__()
            self.pending.append((current, scope))
        def after(current, args, output):
            index = next((i for i in range(len(self.pending) - 1, -1, -1) if self.pending[i][0] is current), None)
            if index is None:
                return
            _, scope = self.pending.pop(index)
            scope.__exit__(RuntimeError if output is None else None, None, None)
            if image_ready and output is not None:
                self.call.image_ready_at = self.call.profiler.clock()
                self.call.encoding_count += 1
                if projected and self.call.layout == "inserted":
                    count = _feature_tokens(output)
                    if count is not None:
                        self.call.tokens["image_input_tokens"] = count
                        self.call.sources["image_input_tokens"] = "projector_output_sequence_shape"
                        self.call._complete_inserted_tokens()
        self.handles.append(module.register_forward_pre_hook(before))
        # always_call ensures phase stacks are closed when a forward fails.
        try:
            handle = module.register_forward_hook(after, always_call=True)
        except TypeError:
            handle = module.register_forward_hook(after)
        self.handles.append(handle)

    def close(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for owner, original, own in reversed(self.method_restores):
            if own:
                owner.encode = original
            else:
                delattr(owner, "encode")
        self.method_restores.clear()
        while self.pending:
            _, scope = self.pending.pop()
            scope.__exit__(exc_type, exc, traceback)

    def __exit__(self, exc_type, exc, traceback):
        self.close(exc_type, exc, traceback)
        return False
