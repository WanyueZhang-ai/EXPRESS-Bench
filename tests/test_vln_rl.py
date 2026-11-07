import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from InternDecision_scripts.rl.environment import HabitatTrainingEnvironment
from InternDecision_scripts.rl.tasks import check_scenes, load_tasks, resolve_scene


class VLNTests(unittest.TestCase):
    def env(self, distance=2.0, max_steps=2):
        env = HabitatTrainingEnvironment([{}], '.', '.', lambda *args: self.fail('QA reward called'), max_steps=max_steps)
        env.is_vln = True
        env.task = {'task_type': 'vln', 'goals': [{'position': [0, 0, 0]}]}
        env.memory.record = lambda *args: None
        env.request = {}
        env.state = {}
        env.step_count = 0
        env.distance = 0
        env.initial_goal_distance = 2
        env.goal_distance = lambda: distance
        env.features = lambda: [0] * 6
        env._observe = lambda *args: None
        env.agent = SimpleNamespace(get_state=lambda: SimpleNamespace(position=np.zeros(3)))
        env.sim = SimpleNamespace(step=lambda action: {})
        env.episode_metrics = {}
        return env

    def test_stop_success_and_no_auxiliary_labels(self):
        env = self.env()
        _, _, reward, done, truncated, info = env.step('stop')
        self.assertEqual(reward, 1)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info['targets'], {})
        self.assertEqual(env.counterfactual_labels(), {})
        self.assertEqual(info['spl'], 1)

    def test_timeout_near_goal_is_failure(self):
        env = self.env(max_steps=1)
        _, _, reward, done, _, info = env.step('move_forward')
        self.assertEqual(reward, 0)
        self.assertTrue(done)
        self.assertTrue(info['budget_exhausted'])
        self.assertFalse(info['success'])

    def test_far_unreachable_and_boundary_stop_fail(self):
        for distance in (3.0, 4.0, float('inf')):
            with self.subTest(distance=distance):
                self.assertEqual(self.env(distance).step('stop')[2], 0)

    def test_intermediate_collision_has_no_dense_reward(self):
        result = self.env().step('move_forward')
        self.assertEqual(result[2], 0)
        self.assertFalse(result[3])

    def test_scene_alias_overlap(self):
        with self.assertRaises(ValueError):
            check_scenes([{'scene_id': '00000-example'}], [{'scene_id': 'hm3d/val/example.basis.glb'}])

    def test_validation_and_scene_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'mp3d' / 'room' / 'room.glb'
            asset.parent.mkdir(parents=True)
            asset.touch()
            self.assertEqual(resolve_scene(root, {'scene_id': 'room'}), asset.resolve())
            task = {'task_type': 'vln', 'split': 'train', 'scene_id': 'room',
                    'instruction': 'Go to the chair', 'start_position': [0, 0, 0],
                    'start_rotation': [0, 0, 0, 1], 'goals': [{'position': [1, 0, 0]}]}
            path = root / 'tasks.json'
            path.write_text(json.dumps([task]))
            self.assertEqual(load_tasks(path, True)[0], task)
            task['start_rotation'] = [1, 1, 1, 1]
            path.write_text(json.dumps([task]))
            with self.assertRaises(ValueError):
                load_tasks(path, True)

    def test_actor_request_excludes_privileged_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.env()
            env.directory = Path(directory)
            env.task['instruction'] = 'Find the chair'
            env.task['goals'] = [{'position': [98, 97, 96]}]
            env.memory.reset('test')
            HabitatTrainingEnvironment._observe(env, {'color_sensor': np.zeros((8, 8, 4), dtype=np.uint8)})
            self.assertEqual(set(env.request['questions']), {'action'})
            text = json.dumps(env.request)
            self.assertIn('Find the chair', text)
            self.assertNotIn('98', text)
            self.assertNotIn('goal_position', text)


if __name__ == '__main__':
    unittest.main()
