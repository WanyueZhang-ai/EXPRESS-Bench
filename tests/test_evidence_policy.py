import unittest

from InternDecision_scripts.evidence_backbone import BackboneScores
from InternDecision_scripts.evidence_policy import EvidenceAwarePolicy


class EvidencePolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = EvidenceAwarePolicy(stop_threshold=0.7, utility_threshold=0.1,
                                          no_progress_patience=3, cost_weight=0.5,
                                          novelty_weight=0.1)
        self.candidates = [
            {"id": "fbe_0", "mode": "fbe", "novelty": 0.8},
            {"id": "goe_0", "mode": "goe", "novelty": 0.2},
        ]
        self.scores = BackboneScores(0.2, {"fbe_0": 0.7, "goe_0": 0.5},
                                     {"fbe_0": 0.2, "goe_0": 0.2}, 5)

    def test_selects_highest_utility_valid_candidate(self):
        decision = self.policy.decide(self.scores, self.candidates, step=2,
                                      max_steps=10, no_progress_steps=0)
        self.assertEqual((decision.action, decision.target_id), ("explore", "fbe_0"))
        self.assertFalse(decision.fallback)

    def test_verifier_and_no_progress_stop_with_reason(self):
        high = BackboneScores(0.9, self.scores.value, self.scores.cost, 5)
        self.assertEqual(self.policy.decide(high, self.candidates, step=0, max_steps=10,
                                            no_progress_steps=0).reason, "evidence_sufficient")
        self.assertEqual(self.policy.decide(self.scores, self.candidates, step=2, max_steps=10,
                                            no_progress_steps=3).reason, "no_progress")

    def test_missing_scores_use_safe_geometry_candidate_without_old_baseline(self):
        decision = self.policy.decide(None, self.candidates, step=0, max_steps=10,
                                      no_progress_steps=0)
        self.assertEqual(decision.target_id, "fbe_0")
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.reason, "model_error_safe_geometry")

    def test_no_candidates_and_budget_stop(self):
        self.assertEqual(self.policy.decide(self.scores, [], step=0, max_steps=10,
                                            no_progress_steps=0).reason, "no_candidates")
        self.assertEqual(self.policy.decide(self.scores, self.candidates, step=9,
                                            max_steps=10, no_progress_steps=0).reason, "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
