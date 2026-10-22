import unittest

from InternDecision_scripts.run_evidence_aware import validate_active_config


class ActiveRunnerTests(unittest.TestCase):
    def test_active_config_disables_privileged_semantics_and_legacy_decision(self):
        cfg = {
            "active_perception": {"enabled": True, "max_candidates": 7},
            "semantic_sensor": True,
            "jev": {"enabled": True},
            "intern_decision": {"enabled": True},
        }
        validate_active_config(cfg)
        self.assertFalse(cfg["semantic_sensor"])
        self.assertFalse(cfg["jev"]["enabled"])
        self.assertFalse(cfg["intern_decision"]["enabled"])

    def test_more_than_seven_candidates_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_active_config({"active_perception": {"enabled": True, "max_candidates": 8}})


if __name__ == "__main__":
    unittest.main()

class ActiveRunnerValidationTests(unittest.TestCase):
    def test_rejects_unknown_goe_provider_and_excess_images(self):
        with self.assertRaises(ValueError):
            validate_active_config({"active_perception": {"enabled": True, "goe_provider": "semantic_regions"}})
        with self.assertRaises(ValueError):
            validate_active_config({"active_perception": {"enabled": True, "max_history_images": 8}})
