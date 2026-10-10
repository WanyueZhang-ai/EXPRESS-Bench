import logging
import tempfile
import unittest
from pathlib import Path


class InternDecisionRepairTests(unittest.TestCase):
    def test_scene_dataset_config_is_resolved_from_hm3d_split(self):
        from InternDecision_scripts.runtime_utils import scene_dataset_config_path

        scene = Path("data/hm3d/train/00000-abc/abc.basis.glb")
        self.assertEqual(
            scene_dataset_config_path(scene),
            Path("data/hm3d/train/hm3d_annotated_basis.scene_dataset_config.json"),
        )

    def test_clean_output_removes_only_known_run_artifacts(self):
        from InternDecision_scripts.runtime_utils import clean_output_dir

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "12").mkdir()
            (output / "12" / "stale.png").write_bytes(b"old")
            (output / "results_12.pkl").write_bytes(b"old")
            (output / "intern_decision.log").write_text("old", encoding="utf-8")
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            clean_output_dir(output)
            self.assertFalse((output / "12").exists())
            self.assertFalse((output / "results_12.pkl").exists())
            self.assertFalse((output / "intern_decision.log").exists())
            self.assertTrue((output / "keep.txt").exists())

    def test_launcher_imports_main_before_configuring_logging(self):
        source = Path("InternDecision_scripts/run_intern_decision.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("from main import main as benchmark_main"),
            source.index("    configure_worker_logging(output_dir"),
        )

    def test_path_increment_uses_observed_translation_only_for_stepped_actions(self):
        from InternDecision_scripts.action_executor import path_increment

        self.assertAlmostEqual(path_increment("move_forward", [0, 0, 0], [0.25, 0, 0], stepped=True), 0.25)
        self.assertEqual(path_increment("move_forward", [0, 0, 0], [0, 0, 0], stepped=True), 0.0)
        self.assertEqual(path_increment("move_forward", [0, 0, 0], [0.25, 0, 0], stepped=False), 0.0)

    def test_trace_includes_object_count_without_serializing_objects(self):
        from InternDecision_scripts.decision_router import RouterDecision, decision_to_trace

        decision = RouterDecision(
            action="fbe",
            target_id="fbe_0",
            fallback=False,
            state={"map": {"goe_candidates": [], "fbe_candidates": [{"id": "fbe_0"}]}, "objects": [{"id": "object_3"}]},
        )
        trace = decision_to_trace(decision, question_index=0, step=1, seed=42)
        self.assertEqual(trace["evidence_summary"], {"object_count": 1})
        self.assertNotIn("objects", trace)

    def test_worker_logger_writes_to_intern_decision_log(self):
        from InternDecision_scripts.run_intern_decision import configure_worker_logging

        with tempfile.TemporaryDirectory() as temp:
            configure_worker_logging(Path(temp), append_trace=False)
            logging.info("diagnostic record")
            for handler in logging.getLogger().handlers:
                handler.flush()
            payload = (Path(temp) / "intern_decision.log").read_text(encoding="utf-8")
        self.assertIn("diagnostic record", payload)
