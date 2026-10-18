"""Reproducible question subsets without importing model/simulator libraries."""
from __future__ import annotations

import json
from pathlib import Path
import random


SELECTION_FILES = ("selected_indices.json", "question_selection.json")


def add_selection_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--indices-file", default=None, help="JSON list of original dataset indices")
    group.add_argument("--sample-size", type=int, default=None, help="randomly select this many questions without replacement")
    parser.add_argument("--sample-seed", type=int, default=42, help="local question-sampling seed (default: 42)")


def load_question_count(dataset_path):
    with Path(dataset_path).expanduser().open(encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise ValueError(f"dataset must contain a JSON list: {dataset_path}")
    return len(questions)


def validate_indices(indices, total_questions):
    if not isinstance(indices, (list, tuple)) or not indices:
        raise ValueError("question indices must be a nonempty list")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError("question indices must be integers (not booleans or fractions)")
    if len(set(indices)) != len(indices):
        raise ValueError("question indices must not contain duplicates")
    if any(index < 0 or index >= total_questions for index in indices):
        raise ValueError(f"question indices must be in [0, {total_questions})")
    return list(indices)


def select_indices(total_questions, *, sample_size=None, sample_seed=42, indices_file=None):
    if not isinstance(total_questions, int) or isinstance(total_questions, bool) or total_questions <= 0:
        raise ValueError("dataset question count must be a positive integer")
    if sample_size is not None and indices_file is not None:
        raise ValueError("--sample-size and --indices-file cannot be combined")
    if indices_file is not None:
        with Path(indices_file).expanduser().open(encoding="utf-8") as handle:
            return validate_indices(json.load(handle), total_questions)
    if sample_size is None:
        return list(range(total_questions))
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or not 1 <= sample_size <= total_questions:
        raise ValueError(f"--sample-size must be between 1 and {total_questions}")
    if not isinstance(sample_seed, int) or isinstance(sample_seed, bool):
        raise ValueError("--sample-seed must be an integer")
    # Isolated RNG: selecting questions cannot alter model/simulator RNG state.
    # Dataset order retains the original scene locality and stable result IDs.
    return sorted(random.Random(sample_seed).sample(range(total_questions), sample_size))


def selection_metadata(indices, total_questions, *, sample_size=None, sample_seed=42, indices_file=None, dataset_path=None):
    return {"schema_version": 1, "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "dataset_question_count": total_questions, "selected_count": len(indices),
            "method": "indices_file" if indices_file is not None else "random_sample" if sample_size is not None else "full_dataset",
            "sample_seed": sample_seed if sample_size is not None else None,
            "indices_file": str(Path(indices_file).expanduser().resolve()) if indices_file is not None else None,
            "indices": list(indices)}


def resolve_cli_selection(cfg, args):
    """Avoid reading the dataset on ordinary config-only dry runs."""
    if args.sample_size is None and args.indices_file is None:
        return None, None
    total = load_question_count(cfg.dataset_path)
    options = {"sample_size": args.sample_size, "sample_seed": args.sample_seed, "indices_file": args.indices_file}
    indices = select_indices(total, **options)
    return indices, selection_metadata(indices, total, dataset_path=cfg.dataset_path, **options)


def save_selection(output_dir, metadata):
    if metadata is None:
        return
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for filename, value in zip(SELECTION_FILES, (metadata["indices"], metadata)):
        path = destination / filename
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
