import unittest

from InternDecision_scripts.geometry_navigation import bounded_waypoint, GeometryNavigationError, yaw_toward


class GeometryNavigationTests(unittest.TestCase):
    def test_waypoint_is_bounded_along_polyline_and_reports_distance(self):
        route = [(0, 0, 0), (2, 0, 0), (2, 3, 0)]
        result = bounded_waypoint(route, max_step_m=3)
        self.assertEqual(result.position, (2.0, 1.0, 0.0))
        self.assertAlmostEqual(result.distance_m, 3.0)
        self.assertFalse(result.reached_target)

    def test_short_route_reaches_target(self):
        result = bounded_waypoint([(0, 0, 0), (1, 0, 0)], max_step_m=3)
        self.assertEqual(result.position, (1.0, 0.0, 0.0))
        self.assertTrue(result.reached_target)

    def test_missing_route_is_explicit_error(self):
        with self.assertRaises(GeometryNavigationError):
            bounded_waypoint([], max_step_m=3)


    def test_heading_faces_travel_and_scan_turns_when_stationary(self):
        self.assertAlmostEqual(yaw_toward((0, 0), (1, 0), current_yaw=0), -1.57079632679)
        self.assertAlmostEqual(yaw_toward((0, 0), (0, 0), current_yaw=0), 1.57079632679)


if __name__ == "__main__":
    unittest.main()
