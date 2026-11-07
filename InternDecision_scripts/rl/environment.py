"""Static Habitat RGB exploration, with private evidence annotations for training.

Training rows may add `evidence_sets` (alternative sets of semantic sensor IDs).
These must be manually verified for the question. They certify visibility only;
attribute/readability/negative-count questions require richer annotations.
"""
import math
from pathlib import Path

from ..harness import EpisodeMemory
from ..evidence_adapters import build_evidence_state, request_with_questions
from ..joint_policy import joint_questions

ACTIONS = ("move_forward", "turn_left", "turn_right", "stop")


class HabitatTrainingEnvironment:
    def __init__(self, tasks, scene_root, output_dir, answer_reward=None, max_steps=64, seed=42):
        if not tasks or max_steps < 1:
            raise ValueError("nonempty tasks and positive max_steps required")
        self.tasks, self.scene_root = tasks, Path(scene_root)
        self.output_dir = Path(output_dir)
        self.answer_reward = answer_reward
        self.max_steps, self.seed = max_steps, seed
        self.index, self.episode = -1, 0
        self.sim = None
        self.memory = EpisodeMemory()

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def reset(self):
        import habitat_sim
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs
        from src.habitat import make_simple_cfg
        self.close()
        self.index = (self.index + 1) % len(self.tasks)
        self.task = self.tasks[self.index]
        self.episode += 1
        scene_id = self.task["scene_id"]
        self.is_vln = self.task.get("task_type") == "vln"
        from .tasks import resolve_scene
        scene = resolve_scene(self.scene_root, self.task)
        if not scene.is_file():
            raise FileNotFoundError(scene)
        settings = dict(scene=str(scene), height=512, width=512, sensor_height=1.5,
                        hfov=90, semantic_sensor=not self.is_vln and bool(self.task.get("evidence_sets")))
        config = make_simple_cfg(settings)
        if self.is_vln:
            config.agents[0].action_space = {
                name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=amount))
                for name, amount in (("move_forward", 0.25), ("turn_left", 15), ("turn_right", 15))
            }
        self.sim = habitat_sim.Simulator(config)
        self.sim.seed(self.seed + self.episode)
        navmesh = scene.with_suffix(".navmesh")
        if navmesh.is_file():
            self.sim.pathfinder.load_nav_mesh(str(navmesh))
        if self.is_vln and not self.sim.pathfinder.is_loaded:
            raise ValueError("VLN requires a loaded navmesh")
        self.agent = self.sim.initialize_agent(0)
        pose = self.agent.get_state()
        pose.position = np.asarray(self.task["start_position"], dtype=float)
        pose.rotation = quat_from_coeffs(np.asarray(self.task["start_rotation"], dtype=float))
        self.agent.set_state(pose)
        self.step_count, self.distance = 0, 0.0
        self.seen = set()
        self.episode_metrics = {}
        if self.is_vln:
            self.initial_goal_distance = self.goal_distance()
            if not math.isfinite(self.initial_goal_distance):
                raise ValueError("VLN start cannot reach any goal")
        self.memory.reset(f"train-{self.episode}")
        self.directory = self.output_dir / f"episode-{self.episode:06d}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._observe(self.sim.get_sensor_observations())
        return self.request, self.features()

    def coverage(self):
        sets = [] if self.task.get("task_type") == "vln" else self.task.get("evidence_sets", [])
        if not sets:
            return None
        return max(len(set(map(int, ids)) & self.seen) / len(set(ids)) for ids in sets if ids)

    def _observe(self, observation):
        from PIL import Image
        image = self.directory / f"{self.step_count:05d}.png"
        Image.fromarray(observation["color_sensor"][:, :, :3]).save(image)
        semantic = observation.get("semantic_sensor")
        if semantic is not None:
            import numpy as np
            self.seen.update(int(value) for value in np.unique(semantic) if int(value) >= 0)
        state = build_evidence_state(
            mode="decision_image", question=self.task["instruction"] if self.is_vln else self.task["question"],
            question_type="navigation" if self.is_vln else self.task["type"],
            history={"step": self.step_count, "max_steps": self.max_steps, "path_length": self.distance},
            perception={}, goe_candidates=[], fbe_candidates=[], image_path=image, step=self.step_count,
        )
        self.state = self.memory.update(state, image)
        questions = joint_questions(self.state, "direct_actions", ACTIONS)
        if self.is_vln:
            questions = {"action": {
                "type": "choice",
                "instructions": "Follow the navigation instruction using current and historical RGB views. Stop only when the instructed destination is reached.",
                "criteria": {"move_forward": "Move forward 0.25 metres", "turn_left": "Turn left 15 degrees",
                             "turn_right": "Turn right 15 degrees", "stop": "Stop navigation at the instructed destination"},
            }}
        self.request = request_with_questions(self.state, questions)

    def goal_distance(self):
        import habitat_sim
        import numpy as np
        distances = []
        for goal in self.task["goals"]:
            path = habitat_sim.ShortestPath()
            path.requested_start = self.agent.get_state().position
            path.requested_end = np.asarray(goal["position"], dtype=float)
            if self.sim.pathfinder.find_path(path) and math.isfinite(path.geodesic_distance):
                distances.append(path.geodesic_distance)
        return min(distances, default=float("inf"))

    def features(self):
        import habitat_sim
        import numpy as np
        distance, reachable = 0.0, 0.0
        if self.task.get("task_type") == "vln":
            raw_distance = self.goal_distance()
            distance, reachable = (min(raw_distance / 20, 1.0), 1.0) if math.isfinite(raw_distance) else (1.0, 0.0)
        elif "goal_position" in self.task:
            path = habitat_sim.ShortestPath()
            path.requested_start = self.agent.get_state().position
            path.requested_end = np.asarray(self.task["goal_position"], dtype=float)
            if self.sim.pathfinder.find_path(path) and math.isfinite(path.geodesic_distance):
                distance, reachable = min(path.geodesic_distance / 20, 1.0), 1.0
        coverage = self.coverage()
        return [self.step_count / self.max_steps, min(self.distance / 20, 1),
                float(coverage or 0), float(coverage is not None), distance, reachable]

    def step(self, action):
        import numpy as np
        from types import SimpleNamespace
        if action not in ACTIONS:
            raise ValueError("invalid action")
        previous_coverage = self.coverage()
        targets = {}
        if previous_coverage is not None:
            targets["sufficiency"] = "yes" if previous_coverage >= 1 else "no"
        self.memory.record(SimpleNamespace(action=action, target_id=None, fallback=False))
        terminated = action == "stop"
        reward = 0.0
        if not terminated:
            before = np.asarray(self.agent.get_state().position).copy()
            observation = self.sim.step(action)
            travelled = float(np.linalg.norm(self.agent.get_state().position - before))
            self.distance += travelled
            self.step_count += 1
            self._observe(observation)
            collision = action == "move_forward" and travelled < 1e-5
            reward = -0.005 - 0.01 * travelled - 0.02 * float(collision)
            coverage = self.coverage()
            if coverage is not None:
                reward += 0.1 * (coverage - previous_coverage)
        budget_end = self.step_count >= self.max_steps
        if self.is_vln:
            done = terminated or budget_end
            reward = 0.0
            if done:
                distance = self.goal_distance()
                success = terminated and math.isfinite(distance) and distance < self.task.get("success_distance", 3.0)
                reward = float(success)
                shortest = self.initial_goal_distance
                self.episode_metrics = {"task_type": "vln", "success": bool(success),
                    "spl": float(success) * (shortest / max(shortest, self.distance) if max(shortest, self.distance) > 0 else 1.0),
                    "goal_distance": distance if math.isfinite(distance) else None,
                    "path_length": self.distance, "steps": self.step_count,
                    "stop_called": terminated, "budget_exhausted": budget_end}
            return self.request, self.features(), reward, done, False, {"targets": {}, **self.episode_metrics}
        if terminated or budget_end:
            # Budget exhaustion is a terminal state of this finite-horizon task.
            quality = float(self.answer_reward(self.task, self.memory.answer_state(self.state)))
            if not math.isfinite(quality) or not 0 <= quality <= 1:
                raise ValueError("answer reward must be finite in [0, 1]")
            grounding = self.coverage()
            reward += quality * (grounding if grounding is not None else 1.0)
            terminated = True
        return self.request, self.features(), reward, terminated, False, {"targets": targets}

    def counterfactual_labels(self):
        """One-step static-scene branch labels without contaminating live history."""
        import numpy as np
        before = self.agent.get_state()
        previous = self.coverage()
        labels = {}
        if previous is None:
            return labels
        for index, action in enumerate(ACTIONS[:-1]):
            try:
                observation = self.sim.step(action)
                seen = self.seen | {int(value) for value in np.unique(observation["semantic_sensor"]) if int(value) >= 0}
                coverage = max(len(set(map(int, ids)) & seen) / len(set(ids))
                               for ids in self.task["evidence_sets"] if ids)
                gain = max(0.0, coverage - previous)
                labels[f"gain_{index}"] = "1" if gain >= 0.75 else "0.5" if gain > 0 else "0"
                fraction = 1 / max(1, self.max_steps - self.step_count)
                labels[f"cost_{index}"] = "1" if fraction >= 0.75 else "0.5" if fraction >= 0.25 else "0"
            finally:
                self.agent.set_state(before)
        labels["sufficiency"] = "yes" if previous >= 1 else "no"
        return labels
