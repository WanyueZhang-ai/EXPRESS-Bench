from types import SimpleNamespace

import pytest
import torch

from src.efficiency import EfficiencyProfiler, _SPAN
from src.efficiency_models import model_call
from src.efficiency_report import read_jsonl


class Vision(torch.nn.Module):
    def forward(self, pixels):
        return torch.ones(1, 4, 3)


class TinyVLM(torch.nn.Module):
    def __init__(self, *, inserted=False, broken=False):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=99)
        if inserted:
            self.vision_backbone = Vision()
            self.projector = torch.nn.Identity()
        else:
            self.visual = Vision()
        self.language_model = torch.nn.Identity()
        self.broken = broken
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **kwargs):
        if self.broken:
            raise RuntimeError("forward failed")
        if hasattr(self, "projector"):
            features = self.projector(self.vision_backbone(pixel_values))
        else:
            features = self.visual(pixel_values)
        return self.language_model(features)


def test_hf_vision_boundary_is_observed_and_expanded_tokens_are_counted(tmp_path):
    model = TinyVLM()
    profiler = EfficiencyProfiler(tmp_path, {"mode": "detailed"})
    with profiler.activate(), profiler.episode(0), profiler.step(0):
        with model_call("qwen", "decision", input_modality="text_image", output_kind="decision", device="cpu") as call:
            with call.observe(model):
                model(torch.tensor([[1, 99, 99, 99, 99, 2, 0]]), attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0]]))
    row = read_jsonl(tmp_path / "efficiency_calls.jsonl")[0]
    assert row["tokens"] == {"text_input_tokens": 2, "image_input_tokens": 4, "effective_input_tokens": 6, "generated_tokens": 0}
    assert row["image_encoding_status"] == "observed"
    assert row["input_to_image_encoded_ms"] + row["image_encoded_to_output_ms"] == pytest.approx(row["wall_ms"])
    assert row["image_encoding_count"] == 1
    assert row["model_forward_count"] == 1
    assert row["stages"]["vision.encode"]["count"] == 1
    assert not model._forward_hooks and not model._forward_pre_hooks


@pytest.mark.parametrize("mode", ["light", "detailed"])
def test_prismatic_counts_projected_positions_without_counting_two_towers_twice(tmp_path, mode):
    model = TinyVLM(inserted=True)
    profiler = EfficiencyProfiler(tmp_path, {"mode": mode})
    with profiler.activate(), profiler.episode(0):
        with model_call("prismatic", "scoring", input_modality="text_image", output_kind="scoring", device="cpu") as call:
            call.set_inputs({"input_ids": [[1, 2, 3]]}, layout="inserted")
            with call.observe(model):
                model(torch.tensor([[1, 2, 3]]))
    row = read_jsonl(tmp_path / "efficiency_calls.jsonl")[0]
    assert row["tokens"]["effective_input_tokens"] == 7
    assert row["tokens"]["image_input_tokens"] == 4
    assert row["tokens"]["generated_tokens"] == 0
    if mode == "detailed":
        assert row["observed_module_paths"]["projector"] == "projector"
        assert row["image_encoding_count"] == 1
    else:
        assert row["input_to_image_encoded_ms"] is None


def test_failed_forward_removes_hooks_and_does_not_poison_next_call(tmp_path):
    model = TinyVLM(broken=True)
    profiler = EfficiencyProfiler(tmp_path, {"mode": "detailed"})
    with profiler.activate(), profiler.episode(0):
        with pytest.raises(RuntimeError, match="forward failed"):
            with model_call("tiny", "decision", input_modality="text_image", output_kind="decision") as call, call.observe(model):
                model(torch.tensor([[1, 99, 2]]))
        assert not model._forward_hooks and not model._forward_pre_hooks
        model.broken = False
        with model_call("tiny", "decision", input_modality="text_image", output_kind="decision") as call, call.observe(model):
            model(torch.tensor([[1, 99, 2]]))
    rows = read_jsonl(tmp_path / "efficiency_calls.jsonl")
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert rows[0]["image_encoding_status"] == "unavailable"
    assert rows[1]["image_encoding_status"] == "observed"
    assert _SPAN.get() is None


def test_checkpoint_backend_encode_is_timed_and_restored(tmp_path):
    from InternDecision_scripts.intern_decision_engine import InternDecisionEngine
    class Backend:
        def __init__(self):
            self.model = TinyVLM()

        def encode(self):
            return torch.tensor([[1, 99, 99, 2]])

        def predict(self):
            self.model(self.encode())
            return {"answers": {"stop": {"type": "noul", "noul": .8}}, "usage": {"output_tokens": 1}}
    backend = Backend()
    engine = InternDecisionEngine.__new__(InternDecisionEngine)
    engine._engine = SimpleNamespace(backend=backend, predict=lambda request: backend.predict())
    engine.checkpoint = tmp_path
    engine.model_kwargs = {"device": "cpu"}
    profiler = EfficiencyProfiler(tmp_path, {"mode": "detailed"})
    with profiler.activate(), profiler.episode(0):
        response = engine.predict({"images": ["frame.png"], "questions": {"stop": {"type": "noul"}}})
    assert response["answers"]["stop"]["noul"] == .8
    assert "encode" not in backend.__dict__
    row = read_jsonl(tmp_path / "efficiency_calls.jsonl")[0]
    assert row["stages"]["input.prepare"]["count"] == 1
    assert row["tokens"]["generated_tokens"] == 0  # decision fields are not generated tokens
    assert row["reported_usage"]["output_tokens"] == 1
    assert row["decision_fields"] == 1


def test_unsupported_adapter_reports_missing_encoding_without_estimating(tmp_path):
    profiler = EfficiencyProfiler(tmp_path, {"mode": "detailed"})
    with profiler.activate(), profiler.episode(0):
        with model_call("opaque", "decision", input_modality="text_image", output_kind="decision") as call, call.observe(object()):
            pass
    row = read_jsonl(tmp_path / "efficiency_calls.jsonl")[0]
    assert row["image_encoding_status"] == "unavailable"
    assert row["input_to_image_encoded_ms"] is None
    assert row["tokens"]["image_input_tokens"] is None


def test_checkpoint_property_failure_does_not_prevent_inference(tmp_path):
    class Adapter:
        @property
        def model(self):
            raise RuntimeError("opaque property")

        def predict(self):
            return "unchanged"
    adapter = Adapter()
    profiler = EfficiencyProfiler(tmp_path, {"mode": "detailed"})
    with profiler.activate(), profiler.episode(0):
        with model_call("opaque", "decision", input_modality="text_image", output_kind="decision") as call, call.observe(adapter):
            assert adapter.predict() == "unchanged"
    row = read_jsonl(tmp_path / "efficiency_calls.jsonl")[0]
    assert row["status"] == "ok"
    assert row["image_encoding_status"] == "unavailable"
    assert row["image_encoding_unavailable_reason"] == "module_discovery_failed:RuntimeError"


@pytest.mark.parametrize("mode", ["light", "detailed"])
def test_qwen_generation_uses_actual_output_ids_and_keeps_answer(tmp_path, monkeypatch, mode):
    import gpt
    class Batch(dict):
        def __getattr__(self, key):
            return self[key]

        def to(self, device):
            return self
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 99, 99, 99, 99, 2]]), attention_mask=torch.ones(1, 6, dtype=torch.long))

        def batch_decode(self, values, **kwargs):
            assert values[0].tolist() == [7, 8]
            return ["yes"]
    class Generator(TinyVLM):
        def generate(self, input_ids, attention_mask=None, streamer=None, **kwargs):
            if streamer:
                streamer.put(input_ids)
            self(input_ids, attention_mask)
            for token in [7, 8]:
                if streamer:
                    streamer.put(torch.tensor([token]))
            if streamer:
                streamer.end()
            return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)
    monkeypatch.setattr(gpt, "_MODEL", Generator())
    monkeypatch.setattr(gpt, "_PROCESSOR", Processor())
    monkeypatch.setattr(gpt, "_MODEL_DEVICE", "cpu")
    monkeypatch.setattr(gpt, "prompt_make", lambda *args: ("system", "question"))
    image = tmp_path / "image.png"
    image.write_bytes(b"image fixture")
    profiler = EfficiencyProfiler(tmp_path, {"mode": mode})
    with profiler.activate(), profiler.episode(0):
        assert gpt.qwen3_vl("answer.txt", "q", str(image), device="cpu") == "yes"
    row = read_jsonl(tmp_path / "efficiency_calls.jsonl")[0]
    assert row["purpose"] == "final_answer"
    assert row["tokens"]["generated_tokens"] == 2
    assert row["tokens"]["image_input_tokens"] == 4
    assert row["generation_tokens_per_second"] > 0
    assert row["ttft_status"] == ("observed" if mode == "detailed" else "not_observed")
