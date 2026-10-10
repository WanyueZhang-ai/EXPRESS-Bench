import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class InternDecisionEvidenceTests(unittest.TestCase):
    def test_candidate_ids_and_prompt_features_are_stable_and_redacted(self):
        from InternDecision_scripts.evidence_adapters import build_evidence_state

        from InternDecision_scripts.evidence_adapters import EvidenceError
        with self.assertRaises(EvidenceError):
            build_evidence_state(
                mode="vlm_state", question="where?", question_type="state",
                history={"steps": [0]},
                perception={"stop_vlm_label": "no", "goal_position": [1, 2, 3]},
                goe_candidates=[{"region": "kitchen", "direction": 1, "goal_position": [9, 9, 9]}],
                fbe_candidates=[{"frontier_count": 4, "geodesic_distance": 2}],
                max_candidates=8,
            )
        state = build_evidence_state(
            mode="vlm_state", question="where?", question_type="state",
            history={"steps": [0]}, perception={"stop_vlm_label": "no"},
            goe_candidates=[{"region": "kitchen", "direction": 1}],
            fbe_candidates=[{"frontier_count": 4}], max_candidates=8,
        )
        self.assertEqual(state["map"]["goe_candidates"][0]["id"], "goe_0")
        self.assertEqual(state["map"]["fbe_candidates"][0]["id"], "fbe_0")
        self.assertNotIn("goal_position", str(state))
        self.assertNotIn("geodesic_distance", str(state))
        self.assertEqual(state["map"]["goe_candidates"][0]["region"], "kitchen")

    def test_object_mode_only_emits_ids_visible_in_current_semantic_frame(self):
        from InternDecision_scripts.evidence_adapters import visible_object_summaries

        visible = [[0, 3], [3, -1]]
        objects = [
            SimpleNamespace(id="object_3", category=SimpleNamespace(name=lambda: "chair"), translation=[1, 2, 3]),
            SimpleNamespace(id="object_8", category=SimpleNamespace(name=lambda: "target"), translation=[8, 8, 8]),
        ]
        rows = visible_object_summaries(visible, objects, agent_position=[1, 1, 1])
        self.assertEqual([row["id"] for row in rows], ["object_3"])
        self.assertEqual(rows[0]["category"], "chair")
        self.assertEqual(rows[0]["agent_relative_center_m"], [0.0, 1.0, 2.0])

    def test_object_mode_accepts_numpy_semantic_sensor_arrays(self):
        from InternDecision_scripts.evidence_adapters import visible_object_summaries

        visible = np.array([[0, 3], [3, -1]], dtype=np.int32)
        objects = [
            SimpleNamespace(id="object_3", category=SimpleNamespace(name=lambda: "chair"), translation=[1, 2, 3]),
            SimpleNamespace(id="object_8", category=SimpleNamespace(name=lambda: "target"), translation=[8, 8, 8]),
        ]
        rows = visible_object_summaries(visible, objects)
        self.assertEqual([row["id"] for row in rows], ["object_3"])

    def test_image_mode_requires_absolute_readable_path(self):
        from InternDecision_scripts.evidence_adapters import build_evidence_state

        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.png"
            image.write_bytes(b"not-a-real-image-but-readable")
            state = build_evidence_state(
                mode="decision_image",
                question="q",
                question_type="state",
                history={},
                perception={},
                goe_candidates=[],
                fbe_candidates=[],
                image_path=image,
            )
            self.assertEqual(state["images"], [str(image.resolve())])


if __name__ == "__main__":
    unittest.main()
