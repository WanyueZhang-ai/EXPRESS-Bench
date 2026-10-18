"""Deterministic data-parallel execution helpers for EXPRESS-Bench.

The runner treats each GPU as an independent worker.  A worker loads its own
Habitat/VLM instance and processes a disjoint list of original dataset indices;
this is data parallelism rather than model parallelism.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class WorkerSpec:
    """The files and device assignment for one worker process."""

    worker_id: int
    gpu_id: int
    indices: List[int]
    pending_indices: List[int]
    output_dir: Path
    indices_file: Path
    progress_file: Path


def split_indices(total_questions: int, num_workers: int) -> List[List[int]]:
    """Split ``range(total_questions)`` into balanced contiguous shards.

    The first ``total_questions % num_workers`` shards receive one additional
    item, so the size difference between any two shards is at most one.  The
    original index is retained, which makes merged results independent of
    worker completion order.
    """

    if isinstance(total_questions, bool) or int(total_questions) != total_questions:
        raise ValueError("total_questions must be an integer")
    if isinstance(num_workers, bool) or int(num_workers) != num_workers:
        raise ValueError("num_workers must be an integer")
    total_questions = int(total_questions)
    num_workers = int(num_workers)
    if total_questions < 0:
        raise ValueError("total_questions must be non-negative")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    base, remainder = divmod(total_questions, num_workers)
    shards: List[List[int]] = []
    cursor = 0
    for worker_id in range(num_workers):
        size = base + (1 if worker_id < remainder else 0)
        shards.append(list(range(cursor, cursor + size)))
        cursor += size
    return shards


def resolve_gpu_ids(
    num_gpus: Optional[int], gpu_ids: Optional[Sequence[int] | str]
) -> List[int]:
    """Resolve and validate the physical GPU IDs used by the workers."""

    parsed: Optional[List[int]]
    if gpu_ids is None:
        parsed = None
    elif isinstance(gpu_ids, str):
        values = [value.strip() for value in gpu_ids.split(",") if value.strip()]
        if not values:
            raise ValueError("--gpus must contain at least one GPU id")
        try:
            parsed = [int(value) for value in values]
        except ValueError as exc:
            raise ValueError("GPU ids must be comma-separated integers") from exc
    else:
        try:
            parsed = [int(value) for value in gpu_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError("GPU ids must be integers") from exc

    if parsed is None:
        count = 1 if num_gpus is None else int(num_gpus)
        if count <= 0:
            raise ValueError("num_gpus must be positive")
        parsed = list(range(count))
    elif num_gpus is not None and int(num_gpus) != len(parsed):
        raise ValueError("num_gpus must equal the number of ids in --gpus")

    if any(gpu_id < 0 for gpu_id in parsed):
        raise ValueError("GPU ids must be non-negative")
    if len(set(parsed)) != len(parsed):
        raise ValueError("GPU ids must be unique")
    return parsed


def write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one worker's current lifecycle/progress state."""

    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["updated_at"] = time.time()
    temporary_path = progress_path.with_name(progress_path.name + f".{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, progress_path)


def read_progress(path: Path) -> Mapping[str, Any]:
    """Read a progress file, returning an explicit unavailable state on a race."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"status": "starting", "completed_questions": 0}
    return value if isinstance(value, Mapping) else {"status": "invalid"}


def build_worker_command(
    *,
    python_executable: str,
    main_script: Path,
    cfg_file: Path,
    indices_file: Path,
    output_dir: Path,
    progress_file: Optional[Path] = None,
    append_trace: bool = False,
    efficiency_mode: Optional[str] = None,
) -> List[str]:
    """Build the subprocess command for one shard."""

    command = [
        str(python_executable),
        str(main_script),
        "--cfg-file",
        str(cfg_file),
        "--indices-file",
        str(indices_file),
        "--output-dir",
        str(output_dir),
    ]
    if progress_file is not None:
        command.extend(["--progress-file", str(progress_file)])
    if append_trace:
        command.append("--append-trace")
    if efficiency_mode is not None:
        command.extend(["--efficiency-mode", efficiency_mode])
    return command


def _completed_episode_indices(worker_dir: Path) -> List[int]:
    """Return question indices with durable per-question or aggregate results."""

    worker_dir = Path(worker_dir)
    completed = []
    aggregate_path = worker_dir / "results.pkl"
    if aggregate_path.exists():
        try:
            with aggregate_path.open("rb") as handle:
                records = pickle.load(handle)
            if isinstance(records, list):
                completed.extend(
                    int(record["question_ind"])
                    for record in records
                    if isinstance(record, Mapping) and isinstance(record.get("question_ind"), int)
                )
        except (OSError, EOFError, pickle.PickleError):
            pass
    for child in worker_dir.iterdir() if worker_dir.exists() else []:
        if not child.is_dir() or not child.name.isdigit():
            continue
        result_path = child / "result.pkl"
        if not result_path.exists():
            continue
        try:
            with result_path.open("rb") as handle:
                record = pickle.load(handle)
            index = record.get("question_ind") if isinstance(record, Mapping) else None
            if isinstance(index, int):
                completed.append(index)
        except (OSError, EOFError, pickle.PickleError):
            continue
    return sorted(set(completed))


def prepare_workers(
    *,
    total_questions: int,
    gpu_ids: Sequence[int],
    output_dir: Path,
    overwrite: bool = False,
    resume: bool = False,
    question_indices: Optional[Sequence[int]] = None,
) -> List[WorkerSpec]:
    """Create shard directories and index manifests before launching workers."""

    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    from src.question_selection import validate_indices
    selected_indices = (list(range(total_questions)) if question_indices is None
                        else validate_indices(question_indices, total_questions))
    output_dir = Path(output_dir)
    shard_root = output_dir / "shards"
    generated_paths = [output_dir / "manifest.json", output_dir / "results.pkl"]
    if resume:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists() or not shard_root.exists():
            raise FileNotFoundError(
                "--resume requires an existing parallel output with manifest.json and shards/"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_gpu_ids = [int(item["gpu_id"]) for item in manifest["workers"]]
            manifest_total = int(manifest["total_questions"])
            manifest_indices = manifest.get("question_indices", list(range(manifest_total)))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid parallel manifest; cannot resume safely") from exc
        if (manifest_total != int(total_questions) or manifest_gpu_ids != [int(gpu) for gpu in gpu_ids]
                or manifest_indices != selected_indices):
            raise ValueError(
                "resume configuration differs from the existing manifest; use the same "
                "dataset size, selected question indices and GPU list"
            )
    elif output_dir.exists() and not overwrite:
        existing_workers = list(shard_root.glob("worker_*")) if shard_root.exists() else []
        existing_files = [path for path in generated_paths if path.exists()]
        if existing_workers or existing_files:
            raise FileExistsError(
                "output directory already contains a parallel run; use --overwrite, "
                "--resume, or choose a new --output-dir"
            )
    if overwrite and shard_root.exists():
        for path in shard_root.glob("worker_*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    if overwrite:
        from src.efficiency import TRACE_FILES, REPORT_FILES
        from src.question_selection import SELECTION_FILES
        for filename in tuple(TRACE_FILES.values()) + REPORT_FILES + SELECTION_FILES:
            path = output_dir / filename
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    shards = [[selected_indices[position] for position in shard]
              for shard in split_indices(len(selected_indices), len(gpu_ids))]
    workers: List[WorkerSpec] = []
    for worker_id, (gpu_id, indices) in enumerate(zip(gpu_ids, shards)):
        worker_dir = shard_root / f"worker_{worker_id:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        full_indices_file = worker_dir / "indices.json"
        full_indices_file.write_text(
            json.dumps(indices, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        completed = set(_completed_episode_indices(worker_dir)) if resume else set()
        invalid_completed = sorted(completed - set(indices))
        if invalid_completed:
            raise ValueError(
                f"worker {worker_id} has results outside its shard: {invalid_completed[:10]}"
            )
        pending_indices = [index for index in indices if index not in completed]
        indices_file = worker_dir / ("pending_indices.json" if resume else "indices.json")
        if resume:
            indices_file.write_text(
                json.dumps(pending_indices, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        worker = WorkerSpec(
            worker_id=worker_id,
            gpu_id=int(gpu_id),
            indices=indices,
            pending_indices=pending_indices,
            output_dir=worker_dir,
            indices_file=indices_file,
            progress_file=worker_dir / "progress.json",
        )
        write_progress(
            worker.progress_file,
            {
                "worker_id": worker.worker_id,
                "gpu_id": worker.gpu_id,
                "status": "completed" if not pending_indices else "queued",
                "phase": "completed" if not pending_indices else "queued",
                "completed_questions": len(completed),
                "total_questions": len(worker.indices),
                "current_question": None,
                "current_step": None,
                "pid": None,
            },
        )
        workers.append(worker)
    manifest = {
        "schema_version": 1,
        "total_questions": int(total_questions),
        "selected_count": len(selected_indices),
        "question_indices": selected_indices,
        "worker_count": len(workers),
        "workers": [
            {
                "worker_id": worker.worker_id,
                "gpu_id": worker.gpu_id,
                "count": len(worker.indices),
                "pending_count": len(worker.pending_indices),
                "indices_file": str(worker.indices_file),
                "output_dir": str(worker.output_dir),
            }
            for worker in workers
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return workers


def _load_result_record(result_path: Path) -> Mapping[str, Any]:
    with result_path.open("rb") as handle:
        record = pickle.load(handle)
    if not isinstance(record, Mapping):
        raise ValueError(f"worker result is not a mapping: {result_path}")
    if not isinstance(record.get("question_ind"), int):
        raise ValueError(f"worker result has an invalid question_ind: {result_path}")
    return record


def _read_worker_results(worker_dir: Path) -> List[Mapping[str, Any]]:
    """Read both aggregate and durable per-question results for resume runs."""

    worker_dir = Path(worker_dir)
    aggregate_path = worker_dir / "results.pkl"
    aggregate_records: List[Mapping[str, Any]] = []
    if aggregate_path.exists():
        with aggregate_path.open("rb") as handle:
            aggregate_records = pickle.load(handle)
        if not isinstance(aggregate_records, list):
            raise ValueError(f"worker result must be a list: {aggregate_path}")
        aggregate_records = [_validate_result_record(record, aggregate_path) for record in aggregate_records]

    episode_records = []
    for child in worker_dir.iterdir() if worker_dir.exists() else []:
        if not child.is_dir() or not child.name.isdigit():
            continue
        result_path = child / "result.pkl"
        if result_path.exists():
            episode_records.append(_load_result_record(result_path))
    if not aggregate_records and not episode_records:
        raise FileNotFoundError(f"worker result is missing: {worker_dir / 'results.pkl'}")

    # The per-question files are authoritative when resuming; collapse the
    # duplicate copy that also appears in a worker's aggregate results.pkl.
    combined = {}
    for record in aggregate_records + episode_records:
        index = int(record["question_ind"])
        if index in combined and combined[index] is not record and episode_records and record not in episode_records:
            raise ValueError(f"duplicate question index in worker result: {index}")
        combined[index] = record
    return list(combined.values())


def _validate_result_record(record: Any, source: Path) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"worker result contains a non-mapping row: {source}")
    if not isinstance(record.get("question_ind"), int):
        raise ValueError(f"worker result has an invalid question_ind: {source}")
    return record


def merge_results(
    shard_dirs: Sequence[Path],
    output_dir: Path,
    *,
    expected_indices: Optional[Iterable[int]] = None,
) -> List[Mapping[str, Any]]:
    """Validate, sort, and write the complete result list."""

    records: List[Mapping[str, Any]] = []
    for shard_dir in shard_dirs:
        records.extend(_read_worker_results(Path(shard_dir)))
    by_index = {}
    duplicates = []
    for record in records:
        index = int(record["question_ind"])
        if index in by_index:
            duplicates.append(index)
        by_index[index] = record
    if duplicates:
        raise ValueError(f"duplicate question indices in worker results: {sorted(set(duplicates))}")

    if expected_indices is not None:
        expected = {int(index) for index in expected_indices}
        actual = set(by_index)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing[:10]}")
            if unexpected:
                details.append(f"unexpected={unexpected[:10]}")
            raise ValueError("worker results do not cover the expected indices (" + ", ".join(details) + ")")

    merged = [by_index[index] for index in sorted(by_index)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.pkl").open("wb") as handle:
        pickle.dump(merged, handle)
    return merged


def merge_jev_traces(shard_dirs: Sequence[Path], output_dir: Path) -> int:
    """Merge Jev and Intern decision JSONL traces deterministically.

    The historical function name is retained for Jev callers. Intern runs use
    the same launcher and receive an additional ``intern_decisions.jsonl``
    merged file; missing trace types are harmless.
    """

    total_records = 0
    for filename in ("jev_decisions.jsonl", "intern_decisions.jsonl"):
        records: List[Mapping[str, Any]] = []
        source_found = False
        for shard_dir in shard_dirs:
            trace_path = Path(shard_dir) / filename
            if not trace_path.exists():
                continue
            source_found = True
            for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid decision trace at {trace_path}:{line_number}") from exc
                if not isinstance(record, Mapping):
                    raise ValueError(f"decision trace row is not an object: {trace_path}:{line_number}")
                records.append(record)

        def sort_key(record: Mapping[str, Any]) -> tuple:
            episode = record.get("episode_id", record.get("question_index"))
            step = record.get("step")
            try:
                episode_key = (0, int(episode))
            except (TypeError, ValueError):
                episode_key = (1, str(episode))
            try:
                step_key = int(step)
            except (TypeError, ValueError):
                step_key = 0
            return episode_key, step_key

        if not source_found:
            continue
        records.sort(key=sort_key)
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        total_records += len(records)
    return total_records


def merge_efficiency_traces(shard_dirs: Sequence[Path], output_dir: Path):
    """Keep original attempts and regenerate a canonical resume-aware report."""
    from src.efficiency import TRACE_FILES
    if not any((Path(directory) / filename).exists() for directory in shard_dirs for filename in TRACE_FILES.values()):
        return None
    from src.efficiency_report import build_report
    return build_report(shard_dirs, output_dir, merge_traces=True)
