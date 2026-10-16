"""Small runtime helpers that do not import Habitat or model libraries."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from src.efficiency import TRACE_FILES, REPORT_FILES


_SCENE_DATASET_CONFIG = "hm3d_annotated_basis.scene_dataset_config.json"
_CLEAN_ROOT_FILES = {
    "intern_decision.log",
    "intern_decisions.jsonl",
    "jev_decisions.jsonl",
    "progress.json",
    "results.pkl",
}
_CLEAN_ROOT_FILES.update(TRACE_FILES.values())
_CLEAN_ROOT_FILES.update(REPORT_FILES)


def scene_dataset_config_path(scene_path: str | Path) -> Path:
    """Return the HM3D dataset config next to a split's scene folders.

    Scene files are laid out as ``.../hm3d/<split>/<scene-id>/<file>.basis.glb``.
    Habitat expects the dataset config in the split directory, two parents above
    the scene file.
    """
    scene = Path(scene_path)
    if len(scene.parents) < 2:
        raise ValueError(f"scene path is missing the split directory: {scene}")
    return scene.parents[1] / _SCENE_DATASET_CONFIG


def clean_output_dir(output_dir: str | Path) -> None:
    """Remove only artifacts produced by an EXPRESS-Bench run.

    This is intentionally opt-in from the launcher. Numeric question folders
    contain images and per-question ``result.pkl`` files, while root-level
    ``results_*.pkl`` files are periodic checkpoints.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for child in list(output.iterdir()):
        if child.is_dir() and re.fullmatch(r"\d+", child.name):
            shutil.rmtree(child)
            continue
        if child.is_file() and (
            child.name in _CLEAN_ROOT_FILES
            or child.name.startswith("results_") and child.suffix == ".pkl"
        ):
            child.unlink()
