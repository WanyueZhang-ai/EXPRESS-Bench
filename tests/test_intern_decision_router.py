import unittest


class FakeEngine:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        return self.response


def _answer(kind, decision, confidence=0.9):
    if kind == "noul":
        return {"type": "noul", "noul": 0.1, "confidence": confidence, "decision": decision, "probabilities": {"no": 0.9, "yes": 0.1}}
    return {"type": "choice", "choice": decision, "decision": decision, "confidence": confidence, "probabilities": {decision: confidence}}


class InternDecisionRouterTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "schema_version": "x",
            "task": {"question": "where?", "question_type": "state"},
            "history": {},
            "perception": {},
            "map": {
                "goe_candidates": [{"id": "goe_0", "region": "kitchen"}],
                "fbe_candidates": [{"id": "fbe_0", "frontier_count": 3}],
            },
        }

    def test_planner_decision_uses_stable_candidate_and_same_step_image(self):
        from InternDecision_scripts.decision_router import InternDecisionRouter

        response = {"answers": {
            "stop": _answer("noul", "no"),
            "mode": _answer("choice", "goe"),
            "target": _answer("choice", "goe_0"),
        }, "model": "fake", "usage": {"input_tokens": 10}}
        engine = FakeEngine(response)
        router = InternDecisionRouter(engine)
        state = dict(self.state)
        state["images"] = ["/tmp/frame.png"]
        decision = router.decide(state, baseline_action="fbe", baseline_target_id="fbe_0")

        self.assertEqual((decision.action, decision.target_id), ("goe", "goe_0"))
        self.assertFalse(decision.fallback)
        self.assertEqual(engine.requests[0]["images"], ["/tmp/frame.png"])
        self.assertNotIn("images", engine.requests[0]["state"])

    def test_stop_threshold_returns_stop(self):
        from InternDecision_scripts.decision_router import InternDecisionRouter

        response = {"answers": {"stop": {"type": "noul", "noul": 0.95, "confidence": 0.95, "probabilities": {"no": 0.05, "yes": 0.95}, "decision": "yes"}}, "model": "fake", "usage": {}}
        decision = InternDecisionRouter(FakeEngine(response)).decide(self.state, baseline_action="fbe", baseline_target_id="fbe_0")
        self.assertEqual(decision.action, "stop")
        self.assertIsNone(decision.target_id)

    def test_invalid_target_falls_back_without_exception(self):
        from InternDecision_scripts.decision_router import InternDecisionRouter

        response = {"answers": {"stop": _answer("noul", "no"), "mode": _answer("choice", "goe"), "target": _answer("choice", "fbe_0")}, "model": "fake", "usage": {}}
        decision = InternDecisionRouter(FakeEngine(response)).decide(self.state, baseline_action="fbe", baseline_target_id="fbe_0")
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.fallback_reason, "invalid_target")
        self.assertEqual((decision.action, decision.target_id), ("fbe", "fbe_0"))

    def test_direct_mode_accepts_whitelist_and_rejects_unknown(self):
        from InternDecision_scripts.decision_router import InternDecisionRouter

        response = {"answers": {"stop": _answer("noul", "no"), "action": _answer("choice", "turn_left")}, "model": "fake", "usage": {}}
        router = InternDecisionRouter(FakeEngine(response), action_mode="direct_actions", direct_actions=["move_forward", "turn_left", "turn_right", "stop"])
        decision = router.decide(self.state, baseline_action="turn_right", baseline_target_id=None)
        self.assertEqual(decision.action, "turn_left")
        self.assertIsNone(decision.target_id)


if __name__ == "__main__":
    unittest.main()
