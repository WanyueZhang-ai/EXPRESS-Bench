import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from src.question_selection import select_indices, save_selection, selection_metadata, validate_indices


def test_ten_question_sample_is_reproducible_and_preserves_rng_state():
    before = random.getstate()
    indices = select_indices(2044, sample_size=10, sample_seed=42)
    assert len(indices) == len(set(indices)) == 10
    assert indices == sorted(indices)
    assert indices == select_indices(2044, sample_size=10, sample_seed=42)
    assert indices != select_indices(2044, sample_size=10, sample_seed=43)
    assert indices != list(range(10))
    assert random.getstate() == before


@pytest.mark.parametrize("size", [0, -1, 21, True, 1.5])
def test_sample_size_never_silently_expands_to_full_dataset(size):
    with pytest.raises(ValueError, match="sample-size"):
        select_indices(20, sample_size=size)


@pytest.mark.parametrize("indices", [[], [0, 0], [True], [1.5], [-1], [20], {"indices": [0]}])
def test_invalid_explicit_indices_fail_before_execution(indices):
    with pytest.raises(ValueError):
        validate_indices(indices, 20)


def test_saved_sample_can_be_reused_by_another_condition(tmp_path):
    indices = select_indices(50, sample_size=10, sample_seed=42)
    save_selection(tmp_path, selection_metadata(indices, 50, sample_size=10, sample_seed=42))
    assert select_indices(50, indices_file=tmp_path / "selected_indices.json") == indices
    metadata = json.loads((tmp_path / "question_selection.json").read_text())
    assert metadata["selected_count"] == 10
    assert metadata["method"] == "random_sample"
    assert metadata["sample_seed"] == 42
    with pytest.raises(ValueError, match="cannot be combined"):
        select_indices(50, sample_size=10, indices_file=tmp_path / "selected_indices.json")


def test_standalone_selector_only_requires_standard_library(tmp_path):
    dataset, output = tmp_path / "data.json", tmp_path / "ten.json"
    dataset.write_text(json.dumps([{}] * 25))
    command = [sys.executable, "-I", "-S", str(Path(__file__).resolve().parents[1] / "tools/select_questions.py"),
               "--dataset", str(dataset), "--output-file", str(output)]
    process = subprocess.run(command, text=True, capture_output=True, check=True)
    assert json.loads(process.stdout)["selected_count"] == 10
    assert json.loads(output.read_text()) == select_indices(25, sample_size=10)


def test_intern_launcher_forwards_only_ten_original_question_ids(tmp_path, monkeypatch):
    from InternDecision_scripts import run_intern_decision as launcher
    from types import SimpleNamespace
    dataset, config, output = tmp_path / "data.json", tmp_path / "config.json", tmp_path / "output"
    dataset.write_text(json.dumps([{}] * 50))
    config.write_text(json.dumps({"dataset_path": str(dataset), "output_dir": str(output)}))
    received = []
    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(main=lambda cfg, **kwargs: received.append(kwargs)))
    monkeypatch.setattr(launcher, "configure_worker_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["launcher", "--cfg-file", str(config), "--sample-size", "10", "--sample-seed", "7"])
    assert launcher.main() == 0
    assert received[0]["question_indices"] == select_indices(50, sample_size=10, sample_seed=7)
    assert json.loads((output / "selected_indices.json").read_text()) == received[0]["question_indices"]


def test_baseline_cli_forwards_ten_question_subset_without_loading_models(tmp_path, monkeypatch):
    import ast
    import logging
    import os
    from src.efficiency import apply_efficiency_cli
    source = Path(__file__).resolve().parents[1] / "main.py"
    parsed = ast.parse(source.read_text())
    entry = next(node for node in parsed.body if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
    dataset, config, output = tmp_path / "data.json", tmp_path / "config.json", tmp_path / "output"
    dataset.write_text(json.dumps([{}] * 50))
    config.write_text(json.dumps({"dataset_path": str(dataset), "output_dir": str(output)}))
    received = []
    monkeypatch.setattr(sys, "argv", ["main.py", "--cfg-file", str(config), "--sample-size", "10", "--sample-seed", "8"])
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: [handler.close() for handler in kwargs.get("handlers", []) if isinstance(handler, logging.FileHandler)])
    namespace = {"__name__": "__main__", "json": json, "os": os, "logging": logging,
                 "apply_efficiency_cli": apply_efficiency_cli, "main": lambda cfg, **kwargs: received.append(kwargs)}
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)
    assert received[0]["question_indices"] == select_indices(50, sample_size=10, sample_seed=8)


def test_parallel_resume_rejects_changed_subset_without_mutating_manifest(tmp_path):
    from Jev_scripts.parallel_runner import prepare_workers
    from Jev_scripts.run_parallel import run_parallel
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    output = tmp_path / "out"
    options = dict(cfg_file=config, output_dir=output, total_questions=50, num_gpus=2,
                   sample_size=10, sample_seed=42, dry_run=True)
    result = run_parallel(**options)
    assert result["total_questions"] == 10
    assert result["dataset_question_count"] == 50
    expected = select_indices(50, sample_size=10, sample_seed=42)
    assert [index for worker in result["workers"] for index in worker.indices] == expected
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="selected question indices"):
        run_parallel(**{**options, "sample_seed": 43, "resume": True})
    assert (output / "manifest.json").read_bytes() == before
    # Omitting the subset on resume is also rejected, rather than running all 50.
    with pytest.raises(ValueError, match="selected question indices"):
        prepare_workers(total_questions=50, gpu_ids=[0, 1], output_dir=output, resume=True)


def test_resumed_progress_counts_old_completed_questions_in_original_subset(tmp_path):
    import pickle
    from Jev_scripts.parallel_runner import prepare_workers, write_progress
    from Jev_scripts.run_parallel import _progress_line
    workers = prepare_workers(total_questions=50, gpu_ids=[0], output_dir=tmp_path, question_indices=[3, 8, 12, 20, 49])
    with (workers[0].output_dir / "results.pkl").open("wb") as handle:
        pickle.dump([{"question_ind": 3}], handle)
    workers = prepare_workers(total_questions=50, gpu_ids=[0], output_dir=tmp_path, question_indices=[3, 8, 12, 20, 49], resume=True)
    write_progress(workers[0].progress_file, {"total_questions": 4, "completed_questions": 1, "status": "running"})
    line = _progress_line(workers, [])
    assert "questions 2/5" in line
    assert "running 2/5" in line
