import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from contextlib import nullcontext

from src.efficiency import EfficiencyProfiler, current_profiler, profiled_run, _SPAN
from src.efficiency_models import model_call
from src.efficiency_report import build_report, read_jsonl, distribution


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_exclusive_accounting_and_exact_token_coverage(tmp_path):
    clock = Clock()
    profiler = EfficiencyProfiler(tmp_path, clock=clock)
    with profiler.activate(), profiler.episode(3, question_type="existence"), profiler.step(0):
        with profiler.span("simulator.observe", category="simulator"):
            clock.advance(.002)
        with model_call("intern_decision", "decision", input_modality="text", output_kind="decision") as call:
            call.set_inputs({"input_ids": [[1, 2, 0]], "attention_mask": [[1, 1, 0]]})
            with profiler.span("vision.fake"):
                clock.advance(.003)
            clock.advance(.002)
        clock.advance(.003)
    step = read_jsonl(tmp_path / "efficiency_steps.jsonl")[0]
    assert step["wall_ms"] == pytest.approx(10)
    assert sum(step["category_wall_ms"].values()) == pytest.approx(10)
    assert step["category_wall_ms"]["decision_model"] == pytest.approx(5)
    assert step["category_wall_ms"]["simulator"] == pytest.approx(2)
    assert step["model_call_count"] == 1
    assert step["tokens"] == {"text_input_tokens": 2, "image_input_tokens": 0, "effective_input_tokens": 2, "generated_tokens": 0}
    assert read_jsonl(tmp_path / "efficiency_calls.jsonl")[0]["wall_ms"] == pytest.approx(5)
    assert _SPAN.get() is None


def test_unknown_token_counts_are_not_zero(tmp_path):
    profiler = EfficiencyProfiler(tmp_path)
    with profiler.activate(), profiler.episode(0), profiler.step(0):
        with model_call("opaque", "decision", input_modality="text_image", output_kind="decision"):
            pass
    summary = build_report([tmp_path], tmp_path)
    tokens = summary["calls"]["tokens"]
    assert tokens["effective_input_tokens"]["total"] is None
    assert tokens["effective_input_tokens"]["missing_calls"] == 1
    assert tokens["generated_tokens"]["total"] == 0
    step = read_jsonl(tmp_path / "efficiency_steps.jsonl")[0]
    assert step["tokens"]["effective_input_tokens"] is None
    assert step["token_coverage_calls"]["effective_input_tokens"] == 0


def test_lifecycle_error_finalizes_step_episode_run_and_preserves_exception(tmp_path):
    @profiled_run
    def run(cfg):
        profiler = current_profiler()
        profiler.begin_episode(1)
        profiler.begin_step(0)
        with model_call("broken", "decision", input_modality="text", output_kind="decision"):
            raise RuntimeError("model failed")
    cfg = {"output_dir": str(tmp_path), "efficiency": {"enabled": True}}
    with pytest.raises(RuntimeError, match="model failed"):
        run(cfg)
    for filename in ("efficiency_calls.jsonl", "efficiency_steps.jsonl", "efficiency_episodes.jsonl", "efficiency_runs.jsonl"):
        record = read_jsonl(tmp_path / filename)[0]
        assert record["status"] == "error"
        assert record["error_type"] == "RuntimeError"
    assert (tmp_path / "efficiency_report.html").exists()
    assert _SPAN.get() is None
    assert not current_profiler().enabled


def test_disabled_run_does_not_create_outputs(tmp_path):
    output = tmp_path / "disabled"
    @profiled_run
    def run(cfg):
        assert not current_profiler().enabled
        return 42
    assert run({"output_dir": str(output)}) == 42
    assert not output.exists()


def test_warmup_labels_are_per_model_purpose_modality(tmp_path):
    profiler = EfficiencyProfiler(tmp_path, {"warmup_calls": 1})
    with profiler.activate(), profiler.episode(0):
        for modality in ("text", "text", "text_image", "text_image"):
            with model_call("intern", "decision", input_modality=modality, output_kind="decision"):
                pass
    rows = read_jsonl(tmp_path / "efficiency_calls.jsonl")
    assert [row["warmup"] for row in rows] == [True, False, True, False]
    assert build_report([tmp_path], tmp_path)["steady_state_calls"]["count"] == 2


def _attempt(path, *, append=False, fail=False, steps=1):
    profiler = EfficiencyProfiler(path, append=append)
    try:
        with profiler.activate(), profiler.episode(7, question_type="state"):
            for step in range(steps):
                with profiler.step(step), model_call("intern", "decision", input_modality="text", output_kind="decision") as call:
                    call.set_inputs({"input_ids": [[1, 2, 3]]})
                    if fail:
                        raise ValueError("interrupted")
    except ValueError:
        pass


def test_resume_merges_one_canonical_attempt_and_retains_failed_cost(tmp_path):
    worker = tmp_path / "worker_00"
    _attempt(worker, fail=True)
    _attempt(worker, append=True, steps=2)
    # Duplicate input directory simulates accidentally passing the same shard.
    output = tmp_path / "merged"
    summary = build_report([worker, worker], output, merge_traces=True)
    assert summary["episodes"]["count"] == 1
    assert summary["steps"]["count"] == 2
    assert summary["calls"]["count"] == 2
    assert summary["diagnostics"]["raw_calls"] == 3
    assert summary["diagnostics"]["failed_episode_attempts"] == 1
    assert summary["diagnostics"]["excluded_call_statistics"]["count"] == 1
    assert len(read_jsonl(output / "efficiency_calls.jsonl")) == 3
    assert summary["calls"]["tokens"]["effective_input_tokens"]["total"] == 6
    # Rebuilding a merged report is idempotent.
    assert build_report([output], output)["calls"] == summary["calls"]


def test_resume_repairs_truncated_tail_without_discarding_valid_rows(tmp_path):
    _attempt(tmp_path)
    path = tmp_path / "efficiency_calls.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"schema_version":')
    _attempt(tmp_path, append=True)
    assert len(read_jsonl(path)) == 2
    assert len(list(tmp_path.glob("*.truncated-*"))) == 1


def test_report_handles_empty_data_and_escapes_html(tmp_path):
    empty = build_report([tmp_path], tmp_path)
    assert empty["steps"]["wall_ms"]["mean"] is None
    profiler = EfficiencyProfiler(tmp_path)
    with profiler.activate(), profiler.episode(0):
        with model_call("</script><img src=x onerror=alert(1)>", "decision", input_modality="text", output_kind="decision"):
            pass
    build_report([tmp_path], tmp_path)
    page = (tmp_path / "efficiency_report.html").read_text()
    assert "</script><img" not in page
    assert "&lt;/script&gt;" in page
    assert "\\u003c/script\\u003e" in page


def test_distribution_uses_valid_values_and_interpolated_percentiles():
    result = distribution([1, 2, 3, 4, None, float("nan")])
    assert result["count"] == 4
    assert result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)


def test_cli_mode_passes_through_parallel_worker_and_cleaner(tmp_path):
    from omegaconf import OmegaConf
    from src.efficiency import apply_efficiency_cli
    from Jev_scripts.parallel_runner import build_worker_command
    from InternDecision_scripts.runtime_utils import clean_output_dir
    cfg = OmegaConf.create({})
    apply_efficiency_cli(cfg, mode="detailed")
    assert cfg.efficiency.enabled and cfg.efficiency.mode == "detailed"
    command = build_worker_command(python_executable="python", main_script=Path("main.py"), cfg_file=Path("config.yaml"),
                                   indices_file=Path("indices.json"), output_dir=tmp_path, efficiency_mode="detailed")
    assert command[-2:] == ["--efficiency-mode", "detailed"]
    (tmp_path / "efficiency_report.html").write_text("generated")
    (tmp_path / "user_notes.txt").write_text("keep")
    clean_output_dir(tmp_path)
    assert not (tmp_path / "efficiency_report.html").exists()
    assert (tmp_path / "user_notes.txt").read_text() == "keep"


def test_jev_retries_are_measured_once_and_never_log_secret(tmp_path):
    from Jev_scripts.jev_client import JevClient, JevTransportError
    clock, attempts = Clock(), []
    def transport(*args):
        attempts.append(1)
        clock.advance(.01)
        if len(attempts) == 1:
            raise JevTransportError("retry")
        return {"model": "fake", "answers": {"stop": {"type": "noul", "noul": .1}},
                "usage": {"input_tokens": 9, "output_tokens": 2}}
    profiler = EfficiencyProfiler(tmp_path, clock=clock)
    client = JevClient(api_key="never-record-this-secret", transport=transport, sleep=clock.advance, backoff_seconds=.5)
    with profiler.activate(), profiler.episode(0), profiler.step(0):
        client.evaluate({"question": "private prompt"}, {"stop": {"type": "noul"}})
    path = tmp_path / "efficiency_calls.jsonl"
    call = read_jsonl(path)[0]
    assert call["retry_count"] == 1
    assert call["wall_ms"] == pytest.approx(520)
    assert call["stages"]["api.retry_wait"]["wall_ms"] == pytest.approx(500)
    assert call["tokens"]["effective_input_tokens"] == 9
    assert "never-record-this-secret" not in path.read_text()
    assert "private prompt" not in path.read_text()


def test_cuda_events_and_process_memory_have_explicit_separate_scopes(tmp_path, monkeypatch):
    from src import efficiency
    events = []
    class Event:
        def record(self):
            events.append("record")
        def synchronize(self):
            events.append("event_sync")
        def elapsed_time(self, end):
            return 7.0
    cuda = SimpleNamespace(synchronize=lambda device: events.append("device_sync"), device=lambda device: nullcontext(),
                           Event=lambda **kwargs: Event(), get_device_name=lambda device: "fake GPU",
                           memory_allocated=lambda device: 100, memory_reserved=lambda device: 200,
                           reset_peak_memory_stats=lambda device: events.append("reset_peak"),
                           max_memory_allocated=lambda device: 150, max_memory_reserved=lambda device: 300)
    monkeypatch.setattr(efficiency, "_cuda", lambda device: cuda if device is not None else None)
    clock = Clock()
    profiler = EfficiencyProfiler(tmp_path, {"mode": "detailed"}, clock=clock)
    with profiler.activate(), profiler.episode(0), profiler.step(0, memory_device="cuda:1"):
        with model_call("tiny", "decision", input_modality="text", output_kind="decision", device="cuda:1"):
            clock.advance(.02)
    row = read_jsonl(tmp_path / "efficiency_calls.jsonl")[0]
    assert row["wall_ms"] == pytest.approx(20)
    assert row["gpu_timeline_ms"] == 7
    assert "event_sync" in events
    step = read_jsonl(tmp_path / "efficiency_steps.jsonl")[0]
    assert step["memory"]["peak_increment_bytes"] == 50
    assert sum(step["category_wall_ms"].values()) == pytest.approx(20)  # GPU ms are never added


@pytest.mark.parametrize("fail_then_resume,sample_size", [(False, None), (True, None), (True, 10)])
def test_parallel_subprocesses_merge_profiles_and_recover_failed_worker(tmp_path, fail_then_resume, sample_size):
    from Jev_scripts.run_parallel import run_parallel
    worker = tmp_path / "worker.py"
    worker.write_text("""
import argparse, json, os, pickle, sys
from pathlib import Path
sys.path.insert(0, REPOSITORY_ROOT)
from src.efficiency import current_profiler, profiled_run
from src.efficiency_models import model_call

parser = argparse.ArgumentParser()
for option in ('cfg-file', 'indices-file', 'output-dir', 'progress-file', 'efficiency-mode'):
    parser.add_argument('--' + option)
parser.add_argument('--append-trace', action='store_true')
args = parser.parse_args()
indices = json.loads(Path(args.indices_file).read_text())
fail = json.loads(Path(args.cfg_file).read_text())['fail']
cfg = {'output_dir': args.output_dir, 'efficiency': {'enabled': True, 'mode': args.efficiency_mode}}

@profiled_run
def run(cfg, append_trace=False):
    profiler = current_profiler()
    results = []
    for index in indices:
        profiler.begin_episode(index)
        profiler.begin_step(0)
        with profiler.span('simulator.action', category='simulator'):
            pass
        with model_call('worker_model', 'decision', input_modality='text', output_kind='decision') as call:
            call.set_inputs({'input_ids': [[1, 2, 3]]})
            if fail and os.environ['CUDA_VISIBLE_DEVICES'] == '1' and index == indices[0]:
                raise RuntimeError('intentional worker failure')
        profiler.end_step()
        profiler.end_episode()
        results.append({'question_ind': index})
    with (Path(args.output_dir) / 'results.pkl').open('wb') as handle:
        pickle.dump(results, handle)

run(cfg, append_trace=args.append_trace)
""".replace("REPOSITORY_ROOT", repr(str(Path(__file__).resolve().parents[1]))))
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"fail": fail_then_resume}))
    output = tmp_path / "parallel"
    selected_count = sample_size or 2
    options = dict(cfg_file=cfg_file, output_dir=output, total_questions=25 if sample_size else 2, gpus="0,1", num_gpus=2,
                   main_script=worker, python_executable=sys.executable, progress_interval=.05, efficiency_mode="detailed",
                   sample_size=sample_size)
    if fail_then_resume:
        with pytest.raises(RuntimeError, match="one or more GPU workers failed"):
            run_parallel(**options)
        partial = json.loads((output / "efficiency_summary.json").read_text())
        assert partial["episodes"]["status_counts"] == {"ok": selected_count // 2, "error": 1}
        assert (output / "efficiency_report.html").exists()
        cfg_file.write_text(json.dumps({"fail": False}))
    result = run_parallel(**options, resume=fail_then_resume)
    assert result["result_count"] == selected_count
    summary = json.loads((output / "efficiency_summary.json").read_text())
    assert summary["episodes"]["count"] == selected_count
    assert summary["calls"]["count"] == selected_count
    assert summary["calls"]["tokens"]["effective_input_tokens"]["total"] == selected_count * 3
    assert summary["steps"]["simulator_transition_count"] == selected_count
    assert {row["mode"] for row in summary["model_groups"]} == {"detailed"}
    assert len(summary["runs"]) == (3 if fail_then_resume else 2)
    assert summary["diagnostics"]["raw_calls"] == selected_count + int(fail_then_resume)
    selected = json.loads((output / "selected_indices.json").read_text())
    assert sorted(row["episode_id"] for row in summary["episode_records"]) == selected
