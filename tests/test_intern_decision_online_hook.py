import unittest


class InternDecisionOnlineHookTests(unittest.TestCase):
    def test_disabled_runtime_does_not_load_model(self):
        from InternDecision_scripts.online_hook import runtime_from_config
        self.assertIsNone(runtime_from_config({"enabled": False}))

    def test_direct_actions_reject_vlm_state_before_model_load(self):
        from InternDecision_scripts.online_hook import runtime_from_config
        with self.assertRaisesRegex(ValueError, "requires decision_image"):
            runtime_from_config({"enabled": True, "action_mode": "direct_actions", "evidence_mode": "vlm_state"})


if __name__ == "__main__":
    unittest.main()
