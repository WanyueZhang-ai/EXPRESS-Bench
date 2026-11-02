import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from InternDecision_scripts.harness import EpisodeMemory
from InternDecision_scripts.joint_policy import JointDecisionRouter, joint_questions


def state(step=0):
    return {"task": {"question": "Where is the chair?", "question_type": "location"},
            "step": step, "objects": [{"id": "chair1", "category": "chair"}],
            "history": {"max_steps": 10}, "map": {"goe_candidates": [], "fbe_candidates": []}}


class MemoryTests(unittest.TestCase):
    def test_isolation_reset_and_deduplication(self):
        memory = EpisodeMemory()
        memory.reset("one")
        first = memory.update(state())
        memory.update(state(1))
        self.assertEqual(len(memory.objects), 1)
        self.assertEqual(first["state_version"], 1)
        self.assertEqual(len(first["memory"]["objects"][0]["frame_ids"]), 1)
        first["memory"]["objects"].clear()
        self.assertEqual(len(memory.objects), 1)
        memory.reset("two")
        self.assertEqual(len(memory.frames), 0)
        self.assertEqual(memory.version, 0)

    def test_idempotence_and_monotonicity(self):
        memory = EpisodeMemory()
        a = memory.update(state(1))
        self.assertEqual(a, memory.update(state(1)))
        with self.assertRaises(ValueError):
            memory.update(state(0))

    def test_bounded_retrieval_and_current_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = EpisodeMemory(max_frames=3, retrieval_frames=2)
            for step in range(5):
                path = Path(directory) / f"{step}.png"
                path.write_bytes(b"image")
                snapshot = memory.update(state(step), path)
            self.assertEqual(len(memory.frames), 3)
            images = memory.answer_state(snapshot)["images"]
            self.assertEqual(len(images), 2)
            self.assertEqual(images[-1], str(path.resolve()))

    def test_privileged_fields_rejected(self):
        snapshot = state()
        snapshot["ground_truth"] = "chair"
        with self.assertRaises(ValueError):
            EpisodeMemory().update(snapshot)

    def test_joint_single_request_and_action(self):
        class Engine:
            calls = 0
            def predict(self, request):
                self.calls += 1
                self.request = request
                return {"answers": {
                    "action": {"type": "choice", "choice": "move_forward", "probabilities": {"move_forward": 0.8}},
                    "sufficiency": {"noul": 0.1},
                }}
        engine = Engine()
        router = JointDecisionRouter(engine, action_mode="direct_actions")
        decision = router.decide(state(), baseline_action="turn_right", baseline_target_id=None)
        self.assertFalse(decision.fallback)
        self.assertEqual(decision.action, "move_forward")
        self.assertEqual(engine.calls, 1)
        self.assertEqual(len(engine.request["questions"]), 8)


class RLTests(unittest.TestCase):
    def test_gae_terminal_and_truncation(self):
        from InternDecision_scripts.rl.trainer import gae
        advantage, returns = gae([1, 1], [0.5, 0.5], [0.5, 2],
                                [False, True], [False, True], gamma=1, lam=1)
        self.assertEqual(returns, [2, 1])
        advantage, _ = gae([1], [0.5], [2], [False], [True], gamma=1)
        self.assertEqual(advantage, [2.5])

    def test_real_gradients_update_policy_and_critic(self):
        import torch
        from torch import nn
        from torch.distributions import Categorical
        from InternDecision_scripts.rl.trainer import PPOTrainer
        from InternDecision_scripts.rl.policy import PrivilegedCritic
        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = nn.Parameter(torch.zeros(2))
            def forward(self, request):
                return {"action": Categorical(logits=self.weights)}, {"action": ["move_forward", "stop"]}
        policy = Policy()
        trainer = PPOTrainer(policy, PrivilegedCritic(), learning_rate=0.01)
        trainer.supervised_update([{"request": {}, "targets": {"action": "stop"}}])
        self.assertGreater(policy.weights[1], policy.weights[0])
        old = policy.weights.detach().clone()
        with torch.no_grad():
            logp = policy({})[0]["action"].log_prob(torch.tensor(1)).item()
        rows = [{"request": {}, "features": [0]*6, "action": "stop", "old_log_probability": logp,
                 "advantage": 1, "target_return": 1, "targets": {"action": "stop"}}]
        loss = trainer.update(rows, epochs=1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(old, policy.weights.detach()))


class IntegrationTests(unittest.TestCase):
    def test_runtime_keeps_history_for_final_vlm(self):
        from InternDecision_scripts.online_hook import InternDecisionRuntime
        from InternDecision_scripts.answer_backend import AnswerBackend
        calls = []
        runtime = InternDecisionRuntime({}, None, None,
            AnswerBackend(qwen_fn=lambda q, t, s: calls.append(s) or "chair"), EpisodeMemory())
        runtime.reset_episode("q0")
        with tempfile.TemporaryDirectory() as directory:
            for step in range(2):
                path = Path(directory) / f"{step}.png"
                path.write_bytes(b"image")
                snapshot = runtime.evidence(question="Where is the chair?", question_type="location",
                    history={}, perception={}, goe_candidates=[], fbe_candidates=[],
                    image_path=path, step=step)
            runtime.answer("Where is the chair?", "location", snapshot)
            self.assertEqual(len(calls[0]["images"]), 2)
            runtime.reset_episode("q1")
            self.assertEqual(len(runtime.memory.frames), 0)

    def test_differentiable_native_token_adapter(self):
        import torch
        from torch import nn
        from InternDecision_scripts.rl.policy import TrainableDecisionPolicy
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = nn.Parameter(torch.zeros(1, 1, 3))
            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.weights)
        backend = SimpleNamespace(model=Network(), tokenizer=SimpleNamespace(
            encode=lambda symbol, **kwargs: [0 if symbol == "A" else 1]))
        backend.encode = lambda request: (
            SimpleNamespace(fields=["action"], symbols={"action": ["A", "B"]}),
            {"input_ids": torch.tensor([[1]])}, torch.tensor([0]))
        policy = TrainableDecisionPolicy(SimpleNamespace(model=SimpleNamespace(backend=backend)))
        distributions, labels = policy({"questions": {"action": {
            "type": "choice", "criteria": {"move_forward": "move", "stop": "stop"}}}})
        (-distributions["action"].log_prob(torch.tensor(1))).backward()
        self.assertIsNotNone(backend.model.weights.grad)
        self.assertEqual(labels["action"], ["move_forward", "stop"])

    def test_counterfactual_restores_pose_and_does_not_change_memory(self):
        import numpy as np
        from InternDecision_scripts.rl.environment import HabitatTrainingEnvironment
        class Agent:
            pose = 0
            def get_state(self):
                return self.pose
            def set_state(self, pose):
                self.pose = pose
        agent = Agent()
        class Simulator:
            def step(self, action):
                agent.pose += 1
                return {"semantic_sensor": np.array([2])}
        env = HabitatTrainingEnvironment.__new__(HabitatTrainingEnvironment)
        env.agent, env.sim = agent, Simulator()
        env.task = {"evidence_sets": [[1, 2]]}
        env.seen, env.step_count, env.max_steps = {1}, 0, 10
        labels = env.counterfactual_labels()
        self.assertEqual(agent.pose, 0)
        self.assertEqual(env.seen, {1})
        self.assertEqual(labels["gain_0"], "0.5")


if __name__ == "__main__":
    unittest.main()
