"""Offline smoke test for all Intern-Decision request/routing contracts.

This command intentionally uses a fake engine. It does not import Habitat,
Transformers, CUDA, or the downloaded checkpoint.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace

from InternDecision_scripts.decision_router import InternDecisionRouter
from InternDecision_scripts.evidence_adapters import build_evidence_state


class FixtureEngine:
    def __init__(self) -> None:
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        questions = request["questions"]
        answers = {
            "stop": {
                "type": "noul", "noul": 0.0, "confidence": 1.0,
                "choice": "no", "decision": "no",
                "probabilities": {"no": 1.0, "yes": 0.0},
            }
        }
        if "action" in questions:
            answers["action"] = {
                "type": "choice", "choice": "turn_right", "decision": "turn_right",
                "confidence": 1.0, "probabilities": {"turn_right": 1.0},
            }
        else:
            candidates = questions.get("target", {}).get("criteria", {})
            target = next(iter(candidates), None)
            answers["mode"] = {
                "type": "choice", "choice": "goe", "decision": "goe",
                "confidence": 1.0, "probabilities": {"goe": 1.0},
            }
            if target:
                answers["target"] = {
                    "type": "choice", "choice": target, "decision": target,
                    "confidence": 1.0, "probabilities": {target: 1.0},
                }
        return {"answers": answers, "model": "fixture", "usage": {"input_tokens": 1}}


def run_smoke() -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="intern_decision_smoke_") as temp:
        image = Path(temp) / "frame.png"
        image.write_bytes(b"fixture")
        for mode in ("vlm_state", "decision_image", "decision_objects"):
            kwargs = {}
            if mode == "decision_image":
                kwargs["image_path"] = image
            if mode == "decision_objects":
                kwargs["objects"] = [{"id": "object_3", "category": "chair", "agent_relative_center_m": [0.0, 1.0, 2.0]}]
            state = build_evidence_state(
                mode=mode,
                question="Which direction should be explored?",
                question_type="state",
                history={"step": 0},
                perception={"stop_vlm_label": "no"},
                goe_candidates=[{"region": "kitchen", "direction": 1}],
                fbe_candidates=[{"frontier_count": 2}],
                **kwargs,
            )
            engine = FixtureEngine()
            decision = InternDecisionRouter(engine).decide(
                state, baseline_action="fbe", baseline_target_id="fbe_0"
            )
            rows.append({"mode": mode, "action": decision.action, "target_id": decision.target_id})

        direct_state = build_evidence_state(
            mode="decision_image", question="Explore?", question_type="state", history={},
            perception={}, goe_candidates=[], fbe_candidates=[], image_path=image,
        )
        engine = FixtureEngine()
        decision = InternDecisionRouter(
            engine, action_mode="direct_actions",
            direct_actions=("move_forward", "turn_left", "turn_right", "stop"),
        ).decide(direct_state, baseline_action="turn_right", baseline_target_id=None)
        rows.append({"mode": "direct_actions", "action": decision.action, "target_id": None})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = run_smoke()
    print(json.dumps({"status": "ok", "cases": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
