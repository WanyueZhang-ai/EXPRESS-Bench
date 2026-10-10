import unittest


class InternDecisionActionTests(unittest.TestCase):
    def test_one_model_choice_maps_to_one_valid_habitat_transition(self):
        from InternDecision_scripts.action_executor import DirectActionExecutor

        calls = []
        executor = DirectActionExecutor(action_space={"move_forward", "turn_left", "turn_right", "stop"}, safe_action="turn_right")
        result = executor.execute("move_forward", lambda action: calls.append(action) or {"action": action})
        self.assertEqual(calls, ["move_forward"])
        self.assertTrue(result.stepped)
        self.assertFalse(result.fallback)

    def test_invalid_or_colliding_forward_does_not_duplicate_forward(self):
        from InternDecision_scripts.action_executor import DirectActionExecutor

        calls = []
        executor = DirectActionExecutor(action_space={"move_forward", "turn_left", "turn_right", "stop"}, safe_action="turn_right")
        result = executor.execute("move_forward", lambda action: calls.append(action) or {"collided": True})
        self.assertEqual(calls, ["move_forward"])
        self.assertTrue(result.collided)
        next_action = executor.safe_action_for_next_cycle(result)
        self.assertEqual(next_action, "turn_right")
        self.assertEqual(calls, ["move_forward"])

    def test_stop_does_not_step(self):
        from InternDecision_scripts.action_executor import DirectActionExecutor

        calls = []
        executor = DirectActionExecutor(action_space={"move_forward", "turn_left", "turn_right", "stop"}, safe_action="turn_right")
        result = executor.execute("stop", lambda action: calls.append(action))
        self.assertFalse(result.stepped)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
