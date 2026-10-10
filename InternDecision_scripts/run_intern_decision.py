"""Run an Intern-Decision EXPRESS-Bench condition.

Examples are documented in ``InternDecision_scripts/README.md``.  This launcher
keeps the existing ``main.py`` episode/evaluation loop and only selects the local
Intern decision backend through configuration.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-cf", "--cfg-file", default="InternDecision_scripts/config_vlm_state.yaml")
    parser.add_argument("--mode", choices=["vlm_state", "decision_image", "decision_objects"], default=None)
    parser.add_argument("--action-mode", choices=["planner_candidates", "direct_actions"], default=None)
    parser.add_argument("--answer-backend", choices=["qwen_vl", "intern_decision"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--indices-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--progress-file", default=None)
    parser.add_argument("--append-trace", action="store_true")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="remove stale numeric question folders and known run artifacts before starting",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def configure_worker_logging(output_dir: Path, *, append_trace: bool = False) -> None:
    """Configure root logging explicitly for one worker process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(output_dir / "intern_decision.log", mode="a" if append_trace else "w"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def main() -> int:
    args = _parse_args()
    if args.smoke:
        from InternDecision_scripts.run_intern_decision_smoke import run_smoke
        print(json.dumps({"status": "ok", "cases": run_smoke()}, ensure_ascii=False, indent=2))
        return 0

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.cfg_file)
    if "intern_decision" not in cfg or cfg.intern_decision is None:
        cfg.intern_decision = OmegaConf.create({})
    cfg.intern_decision.enabled = True
    if args.mode is not None:
        cfg.intern_decision.evidence_mode = args.mode
    if args.action_mode is not None:
        cfg.intern_decision.action_mode = args.action_mode
    if args.answer_backend is not None:
        cfg.intern_decision.answer_backend = args.answer_backend
    if args.checkpoint is not None:
        cfg.intern_decision.checkpoint = args.checkpoint
    if args.device is not None:
        cfg.intern_decision.device = args.device
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    OmegaConf.resolve(cfg)

    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return 0

    if args.clean_output and args.append_trace:
        raise ValueError("--clean-output cannot be combined with --append-trace")

    output_dir = Path(str(cfg.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        from InternDecision_scripts.runtime_utils import clean_output_dir
        clean_output_dir(output_dir)

    from main import main as benchmark_main

    # Importing main loads Prismatic, whose logging configuration replaces
    # existing root handlers. Configure the worker only after that import.
    configure_worker_logging(output_dir, append_trace=args.append_trace)

    question_indices = None
    if args.indices_file:
        with open(args.indices_file, "r", encoding="utf-8") as handle:
            question_indices = json.load(handle)
        if not isinstance(question_indices, list):
            raise ValueError("--indices-file must contain a JSON list")

    benchmark_main(
        cfg,
        question_indices=question_indices,
        progress_path=args.progress_file,
        append_trace=args.append_trace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
