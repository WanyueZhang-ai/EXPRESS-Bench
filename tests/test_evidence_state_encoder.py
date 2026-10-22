import tempfile
import unittest
from pathlib import Path

from PIL import Image

from InternDecision_scripts.evidence_state_encoder import EvidenceStateEncoder


class EvidenceStateEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def frame(self, name, color):
        path = self.root / name
        Image.new("RGB", (16, 16), color).save(path)
        return path

    def test_single_shared_state_contains_bounded_history_and_images(self):
        encoder = EvidenceStateEncoder(
            "Is the tap on?", "state", history_window=2, max_history_images=1
        )
        first = self.frame("first.png", "red")
        second = self.frame("second.png", "blue")
        encoder.observe(step=0, image_path=first, pose=(0, 0, 0),
                        newly_explored_fraction=0.2)
        state = encoder.observe(step=1, image_path=second, pose=(1, 0, 0),
                                action="fbe_0", displacement_m=1.0,
                                newly_explored_fraction=0.1)
        self.assertEqual(state["task"]["question"], "Is the tap on?")
        self.assertEqual(state["observation"]["step"], 1)
        self.assertEqual(state["history"]["actions"], ["fbe_0"])
        self.assertEqual(state["images"], [str(second.resolve()), str(first.resolve())])
        self.assertEqual(state["history"]["no_progress_steps"], 0)

    def test_duplicate_view_without_map_gain_increments_no_progress(self):
        image = self.frame("same.png", "green")
        encoder = EvidenceStateEncoder("Question?", "state", progress_threshold=0.01)
        encoder.observe(step=0, image_path=image, pose=(0, 0, 0),
                        newly_explored_fraction=0.2)
        state = encoder.observe(step=1, image_path=image, pose=(0, 0, 0),
                                action="fbe_0", displacement_m=0.0,
                                newly_explored_fraction=0.0, collided=True)
        self.assertEqual(state["observation"]["view_novelty"], 0.0)
        self.assertEqual(state["history"]["no_progress_steps"], 1)
        self.assertTrue(state["history"]["last_collision"])

    def test_missing_image_and_invalid_numbers_are_rejected(self):
        encoder = EvidenceStateEncoder("Question?", "state")
        with self.assertRaises(ValueError):
            encoder.observe(step=0, image_path=self.root / "missing.png", pose=(0, 0, 0))
        image = self.frame("valid.png", "white")
        with self.assertRaises(ValueError):
            encoder.observe(step=0, image_path=image, pose=(float("nan"), 0, 0))


    def test_image_limit_counts_history_frames_separately_from_current(self):
        encoder = EvidenceStateEncoder("Question?", "state", max_history_images=2)
        images = [self.frame(f"{i}.png", color) for i, color in
                  enumerate(("red", "green", "blue"))]
        for i, image in enumerate(images):
            state = encoder.observe(step=i, image_path=image, pose=(i, 0, 0),
                                    newly_explored_fraction=0.1)
        self.assertEqual(state["images"], [str(images[2].resolve()), str(images[1].resolve()), str(images[0].resolve())])


    def test_movement_without_new_observation_counts_as_no_progress(self):
        first = self.frame("move1.png", "red")
        second = self.frame("move2.png", "blue")
        encoder = EvidenceStateEncoder("Question?", "state", progress_threshold=0.01)
        encoder.observe(step=0, image_path=first, pose=(0, 0, 0),
                        newly_explored_fraction=0.1)
        state = encoder.observe(step=1, image_path=second, pose=(2, 0, 0),
                                action="goe_0", displacement_m=2.0,
                                newly_explored_fraction=0.0)
        self.assertEqual(state["history"]["no_progress_steps"], 1)


if __name__ == "__main__":
    unittest.main()
