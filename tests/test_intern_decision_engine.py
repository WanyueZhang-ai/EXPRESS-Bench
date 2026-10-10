import json
import tempfile
import unittest
from pathlib import Path


class InternDecisionEngineTests(unittest.TestCase):
    def test_engine_loads_checkpoint_relative_inference_without_changing_request(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "inference.py").write_text(
                """
class DecisionEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def predict(self, request):
        return {'answers': {'stop': {'type': 'noul', 'noul': 0.0, 'choice': 'no', 'confidence': 1.0, 'probabilities': {'no': 1.0, 'yes': 0.0}}}, 'usage': {}, 'model': 'fake'}
"""
            )
            from InternDecision_scripts.intern_decision_engine import InternDecisionEngine

            engine = InternDecisionEngine(checkpoint=checkpoint, device="cpu", dtype="float32")
            request = {"state": {"question": "q"}, "questions": {"stop": {"type": "noul"}}}
            original = json.loads(json.dumps(request))
            response = engine.predict(request)

            self.assertEqual(response["model"], "fake")
            self.assertEqual(request, original)
            self.assertEqual(engine.model_kwargs["device"], "cpu")
            self.assertEqual(engine.model_kwargs["dtype"], "float32")

    def test_engine_registers_dynamic_module_before_exec_for_dataclasses(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "inference.py").write_text(
                """
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Metadata:
    value: int = 1

class DecisionEngine:
    def __init__(self, **kwargs):
        self.metadata = Metadata()
    def predict(self, request):
        return {'model': 'fake-dataclass'}
"""
            )
            from InternDecision_scripts.intern_decision_engine import InternDecisionEngine

            engine = InternDecisionEngine(checkpoint=checkpoint, device="cpu", dtype="float32")
            self.assertEqual(engine.model.metadata.value, 1)

    def test_known_upstream_fstring_syntax_is_repaired_in_memory(self):
        from InternDecision_scripts.intern_decision_engine import _repair_known_inference_syntax
        source = "raise ValueError(f'{row.get('id', '<missing id>')}: questions must be a non-empty object')"
        compile(_repair_known_inference_syntax(source), "fixture_inference.py", "exec")

    def test_engine_rejects_missing_local_model(self):
        from InternDecision_scripts.intern_decision_engine import InternDecisionEngine, InternDecisionError

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(InternDecisionError, "checkpoint directory"):
                InternDecisionEngine(checkpoint=Path(temp) / "missing")


if __name__ == "__main__":
    unittest.main()
