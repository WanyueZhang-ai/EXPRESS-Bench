"""Run the evidence-aware EXPRESS exploration path with one model forward per step."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, MutableMapping


def _section(cfg: Any, name: str) -> Any:
    if isinstance(cfg, dict):
        return cfg.setdefault(name, {})
    if name not in cfg or cfg[name] is None:
        cfg[name] = {}
    return cfg[name]


def validate_active_config(cfg: Any) -> Any:
    active = _section(cfg, "active_perception")
    if not bool(active.get("enabled", False)):
        raise ValueError("active_perception.enabled must be true")
    max_candidates = int(active.get("max_candidates", 7))
    if not 1 <= max_candidates <= 7:
        raise ValueError("max_candidates must be 1..7 for one-forward scoring")
    active["max_candidates"] = max_candidates
    provider = str(active.get("goe_provider", "observed_viewpoints"))
    if provider not in {"observed_viewpoints", "none"}:
        raise ValueError("goe_provider must be observed_viewpoints or none")
    active["goe_provider"] = provider
    history_images = int(active.get("max_history_images", 2))
    if not 1 <= history_images <= 3:
        raise ValueError("max_history_images must be 1..3 for the 4-image Qwen answer limit")
    active["max_history_images"] = history_images
    for key in ("max_fbe_candidates", "max_goe_candidates"):
        amount = int(active.get(key, 4 if key == "max_fbe_candidates" else 3))
        if not 0 <= amount <= max_candidates:
            raise ValueError(f"{key} must be 0..max_candidates")
        active[key] = amount
    cfg["semantic_sensor"] = False
    _section(cfg, "jev")["enabled"] = False
    _section(cfg, "intern_decision")["enabled"] = False
    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", default="InternDecision_scripts/config_evidence_aware.yaml")
    parser.add_argument("--indices-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--progress-file")
    parser.add_argument("--append-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.smoke:
        from .run_evidence_aware_smoke import run_smoke
        print(json.dumps(run_smoke(), ensure_ascii=False, indent=2))
        return 0
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.cfg_file)
    cfg.active_perception.enabled = True
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.active_perception.device = args.device
    OmegaConf.resolve(cfg)
    validate_active_config(cfg)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return 0
    indices = None
    if args.indices_file:
        indices = json.loads(Path(args.indices_file).read_text(encoding="utf-8"))
        if not isinstance(indices, list):
            raise ValueError("--indices-file must contain a JSON list")
    output = Path(str(cfg.output_dir)).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(output / "evidence_aware.log", mode="a" if args.append_trace else "w"), logging.StreamHandler()],
        force=True,
    )
    from .active_episode import run_active
    run_active(cfg, question_indices=indices, progress_path=args.progress_file,
               append_trace=args.append_trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
