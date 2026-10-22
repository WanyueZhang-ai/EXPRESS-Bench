import unittest

from InternDecision_scripts.run_evidence_aware_smoke import run_smoke


class ActiveSmokeTests(unittest.TestCase):
    def test_smoke_uses_one_model_call_and_valid_target(self):
        result = run_smoke()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model_calls"], 1)
        self.assertIn(result["target_id"], result["candidate_ids"])


if __name__ == "__main__":
    unittest.main()
