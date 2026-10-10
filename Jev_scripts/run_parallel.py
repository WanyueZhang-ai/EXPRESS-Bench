"""Launch EXPRESS-Bench on multiple GPUs with deterministic question sharding.

Examples
--------
8 GPUs using devices 0 through 7::

    python Jev_scripts/run_parallel.py --cfg-file fine_eqa.yaml \
        --num-gpus 8 --output-dir experiment_result/jev_8gpu

Four selected physical devices::

    python Jev_scripts/run_parallel.py --cfg-file fine_eqa.yaml \
        --num-gpus 4 --gpus 0,2,5,7 --output-dir experiment_result/jev_4gpu
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from .parallel_runner import (
        WorkerSpec,
        build_worker_command,
        merge_jev_traces,
        merge_results,
        prepare_workers,
        read_progress,
        resolve_gpu_ids,
        write_progress,
    )
except ImportError:  # Support ``python Jev_scripts/run_parallel.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Jev_scripts.parallel_runner import (
        WorkerSpec,
        build_worker_command,
        merge_jev_traces,
        merge_results,
        prepare_workers,
        read_progress,
        resolve_gpu_ids,
        write_progress,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_question_count(cfg_file: Path, repo_root: Path) -> int:
    """Read the dataset length from the same OmegaConf used by main.py."""

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "OmegaConf is required to infer the dataset size; pass --total-questions "
            "or install the project environment"
        ) from exc
    cfg = OmegaConf.load(str(cfg_file))
    OmegaConf.resolve(cfg)
    dataset_path = Path(str(cfg.dataset_path))
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path
    with dataset_path.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise ValueError(f"dataset must be a JSON list: {dataset_path}")
    return len(questions)


def _write_parallel_summary(
    output_dir: Path,
    *,
    cfg_file: Path,
    total_questions: int,
    workers: Sequence[WorkerSpec],
    trace_count: int,
    result_count: int,
    metrics: Dict[str, Any],
) -> None:
    summary = {
        "schema_version": 1,
        "cfg_file": str(cfg_file),
        "total_questions": int(total_questions),
        "result_count": int(result_count),
        "trace_count": int(trace_count),
        "worker_count": len(workers),
        "gpu_ids": [worker.gpu_id for worker in workers],
        "metrics": metrics,
    }
    (output_dir / "parallel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _progress_line(
    workers: Sequence[WorkerSpec],
    processes: Sequence[tuple[WorkerSpec, subprocess.Popen[Any], Any]],
) -> str:
    """Format one compact aggregate progress line for the terminal."""

    process_by_worker = {worker.worker_id: process for worker, process, _ in processes}
    completed_total = 0
    question_total = 0
    details = []
    for worker in workers:
        process = process_by_worker.get(worker.worker_id)
        state = read_progress(worker.progress_file)
        completed = int(state.get("completed_questions", 0) or 0)
        total = int(state.get("total_questions", len(worker.indices)) or 0)
        completed_total += completed
        question_total += total
        status = str(state.get("status", "starting"))
        if process is not None and process.poll() is not None and status not in {"completed", "failed"}:
            status = "exited"
        current = state.get("current_question")
        step = state.get("current_step")
        suffix = ""
        if current is not None:
            suffix = f" q={current}"
            if step is not None:
                suffix += f" step={step}"
        details.append(
            f"w{worker.worker_id}:gpu{worker.gpu_id} {status} "
            f"{completed}/{total}{suffix}"
        )
    return (
        f"[parallel] questions {completed_total}/{question_total} | "
        + " | ".join(details)
    )


def run_parallel(
    *,
    cfg_file: Path,
    output_dir: Path,
    num_gpus: Optional[int] = None,
    gpus: Optional[str] = None,
    total_questions: Optional[int] = None,
    main_script: Optional[Path] = None,
    python_executable: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    progress_interval: float = 5.0,
    resume: bool = False,
) -> Dict[str, Any]:
    """Prepare and run one process per selected GPU, then merge outputs."""

    repo_root = _repo_root()
    cfg_file = Path(cfg_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    main_script = (
        Path(main_script).expanduser().resolve()
        if main_script is not None
        else repo_root / "main.py"
    )
    python_executable = python_executable or sys.executable
    if not cfg_file.exists():
        raise FileNotFoundError(f"config file does not exist: {cfg_file}")
    if not main_script.exists():
        raise FileNotFoundError(f"main.py does not exist: {main_script}")

    if total_questions is None:
        total_questions = _load_question_count(cfg_file, repo_root)
    total_questions = int(total_questions)
    progress_interval = float(progress_interval)
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if total_questions <= 0:
        raise ValueError("total_questions must be positive")

    gpu_ids = resolve_gpu_ids(num_gpus, gpus)
    if len(gpu_ids) > total_questions:
        gpu_ids = gpu_ids[:total_questions]
    workers = prepare_workers(
        total_questions=total_questions,
        gpu_ids=gpu_ids,
        output_dir=output_dir,
        overwrite=overwrite,
        resume=resume,
    )

    active_workers = [worker for worker in workers if worker.pending_indices]
    commands = [
        build_worker_command(
            python_executable=python_executable,
            main_script=main_script,
            cfg_file=cfg_file,
            indices_file=worker.indices_file,
            output_dir=worker.output_dir,
            progress_file=worker.progress_file,
            append_trace=resume,
        )
        for worker in active_workers
    ]
    command_by_worker = {worker.worker_id: command for worker, command in zip(active_workers, commands)}
    for worker in workers:
        command = command_by_worker.get(worker.worker_id)
        (worker.output_dir / "command.json").write_text(
            json.dumps(
                {
                    "worker_id": worker.worker_id,
                    "gpu_id": worker.gpu_id,
                    "cuda_visible_devices": str(worker.gpu_id),
                    "command": command,
                    "skipped": command is None,
                    "skip_reason": "all_questions_already_complete" if command is None else None,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if dry_run:
        return {
            "output_dir": str(output_dir),
            "total_questions": total_questions,
            "gpu_ids": gpu_ids,
            "commands": commands,
            "workers": workers,
        }

    processes: List[tuple[WorkerSpec, subprocess.Popen[Any], Any]] = []
    try:
        for worker, command in zip(active_workers, commands):
            env = os.environ.copy()
            # Each child sees its assigned physical device as local cuda:0.
            env["CUDA_VISIBLE_DEVICES"] = str(worker.gpu_id)
            log_handle = (worker.output_dir / "worker_stdout.log").open(
                "a" if resume else "w", encoding="utf-8"
            )
            process = subprocess.Popen(
                command,
                cwd=str(repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((worker, process, log_handle))
            state = dict(read_progress(worker.progress_file))
            state.update({
                "status": "starting",
                "phase": "starting",
                "pid": process.pid,
            })
            write_progress(worker.progress_file, state)
            print(
                f"[parallel] launched worker {worker.worker_id} on GPU {worker.gpu_id}; "
                f"log={worker.output_dir / 'worker_stdout.log'}",
                flush=True,
            )
    except Exception:
        for _, process, log_handle in processes:
            process.terminate()
            log_handle.close()
        raise

    failures = []
    closed_workers = set()
    last_report = 0.0
    try:
        while True:
            now = time.monotonic()
            if now - last_report >= progress_interval:
                print(_progress_line(workers, processes), flush=True)
                last_report = now
            all_finished = True
            for worker, process, log_handle in processes:
                return_code = process.poll()
                if return_code is None:
                    all_finished = False
                    continue
                if worker.worker_id in closed_workers:
                    continue
                closed_workers.add(worker.worker_id)
                log_handle.close()
                if return_code != 0:
                    state = dict(read_progress(worker.progress_file))
                    state.update({
                        "status": "failed",
                        "phase": "failed",
                        "return_code": return_code,
                    })
                    write_progress(worker.progress_file, state)
                    failures.append(
                        {
                            "worker_id": worker.worker_id,
                            "gpu_id": worker.gpu_id,
                            "return_code": return_code,
                            "log": str(worker.output_dir / "worker_stdout.log"),
                        }
                    )
                elif read_progress(worker.progress_file).get("status") != "completed":
                    state = dict(read_progress(worker.progress_file))
                    state.update({
                        "status": "exited",
                        "phase": "exited",
                        "return_code": return_code,
                    })
                    write_progress(worker.progress_file, state)
            if all_finished:
                break
            time.sleep(min(1.0, progress_interval))
    except BaseException:
        for worker, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
            if worker.worker_id not in closed_workers:
                log_handle.close()
        raise
    print(_progress_line(workers, processes), flush=True)
    if failures:
        raise RuntimeError(
            "one or more GPU workers failed; shard outputs were kept: "
            + json.dumps(failures, ensure_ascii=False)
        )

    shard_dirs = [worker.output_dir for worker in workers]
    merged = merge_results(
        shard_dirs,
        output_dir,
        expected_indices=range(total_questions),
    )
    trace_count = merge_jev_traces(shard_dirs, output_dir)
    try:
        from Jev_scripts.run_jev_replay import express_metrics

        metrics = express_metrics(merged)
    except Exception:
        # Metrics are useful metadata, but a result merge should not be lost if
        # an optional metric field is malformed in an otherwise complete run.
        metrics = {}
    _write_parallel_summary(
        output_dir,
        cfg_file=cfg_file,
        total_questions=total_questions,
        workers=workers,
        trace_count=trace_count,
        result_count=len(merged),
        metrics=metrics,
    )
    return {
        "output_dir": str(output_dir),
        "total_questions": total_questions,
        "gpu_ids": gpu_ids,
        "result_count": len(merged),
        "trace_count": trace_count,
        "metrics": metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EXPRESS-Bench with one independent worker per GPU."
    )
    parser.add_argument(
        "-cf", "--cfg_file", "--cfg-file", required=True, help="OmegaConf config path"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="number of GPUs/workers (default: one; inferred from --gpus when supplied)",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated physical GPU ids, for example 0,2,5,7",
    )
    parser.add_argument(
        "--output-dir", required=True, help="root directory for shards and merged results"
    )
    parser.add_argument(
        "--total-questions",
        type=int,
        default=None,
        help="optional dataset size override, mainly useful for smoke tests",
    )
    parser.add_argument(
        "--main-script", default=None, help="override the repository main.py path"
    )
    parser.add_argument(
        "--python-executable", default=sys.executable, help="Python executable for workers"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove generated worker shard directories before starting",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-question results and run only missing indices",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="prepare shards and print commands without launching"
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="seconds between aggregate progress lines (default: 5)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_parallel(
            cfg_file=Path(args.cfg_file),
            output_dir=Path(args.output_dir),
            num_gpus=args.num_gpus,
            gpus=args.gpus,
            total_questions=args.total_questions,
            main_script=Path(args.main_script) if args.main_script else None,
            python_executable=args.python_executable,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            progress_interval=args.progress_interval,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"parallel run failed: {exc}", file=sys.stderr)
        return 1
    printable = dict(result)
    printable.pop("workers", None)
    if printable.get("commands") is None:
        printable.pop("commands", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
