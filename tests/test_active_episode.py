import unittest
import numpy as np

from InternDecision_scripts.active_episode import prepare_candidate_snapshot, exploration_gain, safe_model_text


class ActiveEpisodeTests(unittest.TestCase):
    def test_snapshot_contains_executable_fbe_and_goe_without_semantics(self):
        free = np.zeros((9, 9), dtype=bool)
        observed = np.zeros((9, 9), dtype=bool)
        free[2:7, 2:7] = True
        observed[2:7, 2:7] = True
        rows, generator = prepare_candidate_snapshot(
            free=free, observed=observed, current_xy=(4.0, 4.0),
            origin_xy=(0.0, 0.0), voxel_size=1.0, floor_height=0.0,
            path_checker=lambda point: True,
            max_candidates=4, max_fbe=2, max_goe=2,
        )
        self.assertTrue(any(row["mode"] == "fbe" for row in rows))
        self.assertTrue(any(row["mode"] == "goe" for row in rows))
        self.assertTrue(all("goal_position" not in row for row in rows))
        self.assertEqual(len(rows), len({row["id"] for row in rows}))
        self.assertEqual(len(generator.target_point(rows[0]["id"])), 3)


    def test_exploration_gain_counts_new_observed_cells(self):
        before = np.array([[True, False], [False, False]])
        after = np.array([[True, True], [False, False]])
        self.assertAlmostEqual(exploration_gain(before, after), 0.5)


    def test_empty_map_has_rotational_scan_candidate(self):
        empty = np.zeros((5, 5), dtype=bool)
        rows, generator = prepare_candidate_snapshot(
            free=empty, observed=empty, current_xy=(2.0, 2.0),
            origin_xy=(0.0, 0.0), voxel_size=1.0, floor_height=0.0,
            path_checker=lambda point: True, max_candidates=2,
            max_fbe=1, max_goe=1,
        )
        self.assertEqual([row["id"] for row in rows], ["fbe_0"])
        self.assertEqual(generator.target_point("fbe_0"), (2.0, 2.0, 0.0))


    def test_failed_answer_call_returns_serializable_status(self):
        def broken(*args):
            raise RuntimeError("model unavailable")
        text, status = safe_model_text(broken, "prompt", "Q?", ["frame.png"], "cuda:0")
        self.assertEqual((text, status), ("", "RuntimeError"))


if __name__ == "__main__":
    unittest.main()

class ActiveEpisodeContractTests(unittest.TestCase):
    def test_goe_provider_can_be_disabled(self):
        free = np.ones((7, 7), dtype=bool)
        observed = np.ones((7, 7), dtype=bool)
        rows, _ = prepare_candidate_snapshot(
            free=free, observed=observed, current_xy=(3.0, 3.0),
            origin_xy=(0.0, 0.0), voxel_size=1.0, floor_height=0.0,
            path_checker=lambda point: True, goe_provider="none",
        )
        self.assertFalse(any(row["mode"] == "goe" for row in rows))

    def test_eac_error_is_valid_zero_mark(self):
        from InternDecision_scripts.active_episode import normalize_eac
        self.assertEqual(normalize_eac("", "RuntimeError"), ("0,0", "RuntimeError"))
        self.assertEqual(normalize_eac("bad answer", "ok"), ("0,0", "invalid_format"))
        self.assertEqual(normalize_eac("1,5", "ok"), ("1,5", "ok"))

    def test_pathfinder_without_points_is_not_executable(self):
        from InternDecision_scripts.active_episode import route_points
        class FakePath:
            points = []
        self.assertIsNone(route_points(FakePath()))
