import unittest

from InternDecision_scripts.geometry_candidates import (
    GeometryCandidateGenerator, CandidateMappingError,
)


class GeometryCandidateTests(unittest.TestCase):
    def test_candidates_have_stable_ids_and_separate_executor_targets(self):
        generator = GeometryCandidateGenerator(max_candidates=3)
        rows = generator.build(
            fbe=[{"point": [1.0, 2.0, 0.0], "distance_m": 2.0},
                 {"point": [2.0, 3.0, 0.0], "distance_m": 3.0}],
            goe=[{"point": [3.0, 4.0, 0.0], "distance_m": 1.0}],
            path_checker=lambda point: True,
        )
        self.assertEqual([row["id"] for row in rows], ["fbe_0", "fbe_1", "goe_0"])
        self.assertNotIn("executor_index", rows[0])
        self.assertEqual(generator.executor_target("goe_0"), ("goe", 0))

    def test_unreachable_and_stale_candidates_are_rejected(self):
        generator = GeometryCandidateGenerator(max_candidates=4)
        rows = generator.build(
            fbe=[{"point": [1, 2, 0], "distance_m": 1}, {"point": [2, 2, 0], "distance_m": 2}],
            goe=[], path_checker=lambda point: point[0] != 1,
        )
        self.assertEqual([row["id"] for row in rows], ["fbe_0"])
        with self.assertRaises(CandidateMappingError):
            generator.executor_target("fbe_1")

    def test_geometry_fields_reject_privileged_values(self):
        generator = GeometryCandidateGenerator()
        with self.assertRaises(ValueError):
            generator.build(fbe=[{"point": [1, 2, 0], "goal_position": [9, 9, 9]}], goe=[], path_checker=lambda p: True)


if __name__ == "__main__":
    unittest.main()
