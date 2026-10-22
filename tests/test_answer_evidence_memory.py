import unittest
from pathlib import Path

from gpt import normalize_image_paths


class AnswerEvidenceMemoryTests(unittest.TestCase):
    def test_multiple_paths_are_normalized_in_order_and_bounded(self):
        paths = ["/tmp/current.png", "/tmp/past.png", "/tmp/older.png"]
        self.assertEqual(normalize_image_paths(paths, max_images=2),
                         [Path(paths[0]), Path(paths[1])])

    def test_single_path_compatibility(self):
        self.assertEqual(normalize_image_paths("/tmp/current.png"),
                         [Path("/tmp/current.png")])

    def test_empty_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_image_paths([], max_images=2)


if __name__ == "__main__":
    unittest.main()
