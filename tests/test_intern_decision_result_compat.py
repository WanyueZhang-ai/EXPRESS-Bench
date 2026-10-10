import pickle
import tempfile
import unittest
from pathlib import Path


class InternDecisionResultCompatibilityTests(unittest.TestCase):
    def test_intern_result_keeps_existing_episode_contract(self):
        required = {
            "question_ind": 0,
            "path": {"step_0": {"pts": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}},
            "gen_answer": "yes",
            "EAC": "1.0,5",
            "cnt_step": 0,
            "goal_dis": 1.0,
            "path_len": 1.0,
            "intern_decisions": [],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.pkl"
            with path.open("wb") as handle:
                pickle.dump(required, handle)
            with path.open("rb") as handle:
                restored = pickle.load(handle)
        self.assertEqual(set(required), set(restored))
        self.assertIsInstance(restored["gen_answer"], str)
        self.assertIsInstance(restored["intern_decisions"], list)
        self.assertIsInstance(restored["path"], dict)

    def test_dual_answer_failure_still_has_serializable_answer_status(self):
        result = {
            "question_ind": 0,
            "gen_answer": "",
            "answer_status": "error",
            "answer_error_type": "RuntimeError",
            "intern_decisions": [],
        }
        payload = pickle.dumps(result)
        restored = pickle.loads(payload)
        self.assertEqual(restored["answer_status"], "error")
        self.assertEqual(restored["gen_answer"], "")


if __name__ == "__main__":
    unittest.main()
