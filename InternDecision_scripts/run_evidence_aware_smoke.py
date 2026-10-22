"""No-GPU/no-Habitat contract smoke for the active-perception controller."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from .active_perception_hook import ActivePerceptionController


class SmokeEngine:
    def __init__(self):
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        answers = {"verifier": {"type": "noul", "noul": 0.1,
                                "probabilities": {"yes": 0.1, "no": 0.9}}}
        for field in request["questions"]:
            if field != "verifier":
                answers[field] = {"type": "score", "score": 0.7,
                                  "probabilities": {"0.5": 0.3, "1": 0.7}}
        return {"answers": answers, "usage": {"input_tokens": 50},
                "timing": {"inference_ms": 1.0}}


def run_smoke() -> dict:
    with tempfile.TemporaryDirectory(prefix="express_active_smoke_") as temp:
        image = Path(temp) / "frame.png"
        Image.new("RGB", (16, 16), "blue").save(image)
        engine = SmokeEngine()
        controller = ActivePerceptionController(
            "What is visible?", "object", engine=engine, max_candidates=2,
            max_steps=5,
        )
        result = controller.step(
            step=0, image_path=image, pose=(0, 0, 0),
            candidates=[{"id": "fbe_0", "mode": "fbe", "novelty": 1.0}],
            newly_explored_fraction=0.1,
        )
        assert result.decision.action == "explore"
        assert engine.calls == 1
        return {"status": "ok", "model_calls": engine.calls,
                "decision_fields": result.trace["decision_fields"],
                "target_id": result.decision.target_id,
                "candidate_ids": result.trace["candidate_ids"]}
