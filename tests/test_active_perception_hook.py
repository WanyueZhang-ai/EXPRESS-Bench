import tempfile
import unittest
from pathlib import Path

from PIL import Image

from InternDecision_scripts.active_perception_hook import ActivePerceptionController


class FakeEngine:
    def __init__(self):
        self.calls = []

    def predict(self, request):
        self.calls.append(request)
        answers = {"verifier": {"type": "noul", "noul": 0.1,
                                "probabilities": {"yes": 0.1, "no": 0.9}}}
        for name in request["questions"]:
            if name == "verifier":
                continue
            answers[name] = {"type": "score", "score": 0.8,
                             "probabilities": {"0.75": 0.2, "1": 0.8}}
        return {"answers": answers, "usage": {"input_tokens": 10},
                "timing": {"inference_ms": 3}}


class ActiveHookTests(unittest.TestCase):
    def test_one_step_has_one_forward_and_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.png"
            Image.new("RGB", (8, 8), "red").save(image)
            engine = FakeEngine()
            controller = ActivePerceptionController(
                "Where is the object?", "location", engine=engine,
                max_candidates=2, max_steps=4,
            )
            result = controller.step(
                step=0, image_path=image, pose=(0, 0, 0),
                candidates=[{"id": "fbe_0", "mode": "fbe", "novelty": 1.0}],
            )
            self.assertEqual(len(engine.calls), 1)
            self.assertEqual(result.decision.action, "explore")
            self.assertEqual(result.decision.target_id, "fbe_0")
            self.assertEqual(result.trace["decision_fields"], 3)
            self.assertEqual(result.trace["model_calls"], 1)
            self.assertEqual(result.trace["candidate_ids"], ["fbe_0"])


if __name__ == "__main__":
    unittest.main()

class ForcedStopTests(unittest.TestCase):
    def test_no_progress_forced_stop_skips_model(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.png"
            Image.new("RGB", (8, 8), "blue").save(image)
            engine = FakeEngine()
            controller = ActivePerceptionController(
                "What color?", "attribute", engine=engine,
                max_steps=4, no_progress_patience=1,
            )
            result = controller.step(
                step=0, image_path=image, pose=(0, 0, 0),
                candidates=[{"id": "fbe_0", "mode": "fbe"}],
                newly_explored_fraction=0.0,
            )
            self.assertEqual(result.decision.reason, "no_progress")
            self.assertEqual(result.trace["model_calls"], 0)
            self.assertEqual(len(engine.calls), 0)

class CalibrationTraceTests(unittest.TestCase):
    def test_trace_keeps_safe_raw_probabilities(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.png"
            Image.new("RGB", (8, 8), "red").save(image)
            controller = ActivePerceptionController("What is visible?", "object", engine=FakeEngine(), max_steps=3)
            result = controller.step(
                step=0, image_path=image, pose=(0, 0, 0),
                candidates=[{"id": "fbe_0", "mode": "fbe"}],
                newly_explored_fraction=0.2,
            )
            self.assertIn("verifier", result.trace["raw_probabilities"])
            self.assertIn("value_fbe_0", result.trace["raw_probabilities"])
