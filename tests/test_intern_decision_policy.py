import unittest


class InternDecisionPolicyTests(unittest.TestCase):
    def test_planner_modes_keep_vlm_tsdf_candidate_generation(self):
        from InternDecision_scripts.runtime_policy import use_vlm_planner_perception

        self.assertTrue(use_vlm_planner_perception(True, "planner_candidates"))
        self.assertTrue(use_vlm_planner_perception(True, "planner_candidates", evidence_mode="decision_image"))
        self.assertTrue(use_vlm_planner_perception(True, "planner_candidates", evidence_mode="decision_objects"))
        self.assertFalse(use_vlm_planner_perception(True, "direct_actions", evidence_mode="decision_image"))
