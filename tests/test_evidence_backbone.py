import unittest

from InternDecision_scripts.evidence_backbone import EvidenceAwareDecisionBackbone


class FakeEngine:
    def __init__(self):
        self.calls = []

    def predict(self, request):
        self.calls.append(request)
        answers = {}
        for name, question in request["questions"].items():
            if name == "verifier":
                answers[name] = {"type": "noul", "noul": 0.2,
                                 "probabilities": {"yes": 0.2, "no": 0.8}}
            else:
                answers[name] = {"type": "score", "score": 0.7,
                                 "probabilities": {"0": 0.0, "1": 1.0}}
        return {"answers": answers, "usage": {"input_tokens": 100},
                "timing": {"inference_ms": 50}}


class EvidenceBackboneTests(unittest.TestCase):
    def test_all_fields_share_one_image_state_and_forward(self):
        engine = FakeEngine()
        backbone = EvidenceAwareDecisionBackbone(engine, max_candidates=7)
        state = {"task": {"question": "Where is the mug?"},
                 "images": ["/tmp/current.png", "/tmp/past.png"],
                 "candidates": [{"id": "fbe_0", "mode": "fbe"},
                                {"id": "goe_0", "mode": "goe"}]}
        result = backbone.score(state)
        self.assertEqual(len(engine.calls), 1)
        request = engine.calls[0]
        self.assertEqual(request["images"], state["images"])
        self.assertNotIn("images", request["state"])
        self.assertEqual(set(request["questions"]),
                         {"verifier", "value_fbe_0", "cost_fbe_0",
                          "value_goe_0", "cost_goe_0"})
        self.assertEqual(result.value["fbe_0"], 0.7)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.decision_fields, 5)

    def test_checkpoint_field_limit_caps_candidate_count(self):
        with self.assertRaises(ValueError):
            EvidenceAwareDecisionBackbone(FakeEngine(), max_candidates=8)

    def test_missing_score_is_explicit_error(self):
        class Broken(FakeEngine):
            def predict(self, request):
                return {"answers": {"verifier": {"type": "noul", "noul": 0.1}}}
        with self.assertRaises(ValueError):
            EvidenceAwareDecisionBackbone(Broken()).score(
                {"images": [], "candidates": [{"id": "fbe_0"}]})


if __name__ == "__main__":
    unittest.main()

class ScoreQuestionSemanticsTests(unittest.TestCase):
    def test_value_and_cost_criteria_have_distinct_meanings(self):
        backbone = EvidenceAwareDecisionBackbone(FakeEngine())
        questions = backbone._questions(["fbe_0"])
        value_text = " ".join(questions["value_fbe_0"]["criteria"].values()).lower()
        cost_text = " ".join(questions["cost_fbe_0"]["criteria"].values()).lower()
        self.assertIn("evidence", value_text)
        self.assertIn("travel", cost_text)
        self.assertNotEqual(value_text, cost_text)

class RawCalibrationTests(unittest.TestCase):
    def test_traceable_probabilities_include_verifier_and_score_fields(self):
        result = EvidenceAwareDecisionBackbone(FakeEngine()).score({
            "images": [], "candidates": [{"id": "fbe_0", "mode": "fbe"}],
        })
        self.assertEqual(result.raw_probabilities["verifier"]["yes"], 0.2)
        self.assertIn("value_fbe_0", result.raw_probabilities)
        self.assertIn("cost_fbe_0", result.raw_probabilities)
