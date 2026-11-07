"""Run seed collection, supervised initialization, or on-policy Habitat training."""
import argparse
import json
from pathlib import Path
import random
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["collect", "sft", "ppo"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", required=True, help="Training-only JSON list. Never pass benchmark test questions.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--holdout-tasks", required=True, help="Held-out benchmark JSON used only for scene overlap checks")
    parser.add_argument("--output", required=True, help="New run directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--answer-device", default="cuda:1")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed-data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import torch
    from ..intern_decision_engine import InternDecisionEngine
    from .policy import TrainableDecisionPolicy, PrivilegedCritic
    from .trainer import PPOTrainer
    from .environment import HabitatTrainingEnvironment, ACTIONS
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from .tasks import load_tasks, check_scenes
    tasks = load_tasks(args.tasks, training=True)
    held_out = load_tasks(args.holdout_tasks, training=False)
    check_scenes(tasks, held_out)
    if args.phase == "collect" and any(task.get("task_type") == "vln" for task in tasks):
        raise ValueError("VLN collect needs an aligned expert teacher; use prepared action-only seed JSONL for sft, or ppo with an initialized policy")
    (output / "run.json").write_text(json.dumps(vars(args), indent=2))

    def answer_reward(task, state):
        from gpt import qwen3_vl
        response = qwen3_vl("prompt/answer.txt", f"Q: {task['question']}\nA: ",
                            state.get("images"), device=args.answer_device)
        # Deterministic exact-match starter reward. Open-ended training requires a
        # validated semantic scorer, supplied through the environment callback API.
        normalize = lambda text: " ".join(str(text).lower().strip().rstrip(".").split())
        return float(normalize(response.removeprefix("A:")) == normalize(task["answer"]))

    env = HabitatTrainingEnvironment(tasks, args.scene_root, output / "observations", answer_reward,
                                     max_steps=args.max_steps, seed=args.seed)
    try:
        if args.phase == "collect":
            with (output / "seed.jsonl").open("w") as stream:
                for _ in range(args.updates):
                    request, _ = env.reset()
                    done = False
                    while not done:
                        targets = env.counterfactual_labels()
                        if not targets:
                            raise ValueError("seed collection requires verified evidence_sets")
                        gains = [float(targets[f"gain_{i}"]) for i in range(3)]
                        action = "stop" if targets["sufficiency"] == "yes" else ACTIONS[max(range(3), key=lambda i: (gains[i], random.random()))]
                        targets["action"] = action
                        stream.write(json.dumps({"request": request, "targets": targets}) + "\n")
                        request, _, _, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
            return
        engine = InternDecisionEngine(args.checkpoint, device=args.device, temperature=1.0)
        policy = TrainableDecisionPolicy(engine)
        trainer = PPOTrainer(policy, PrivilegedCritic())
        examples = []
        if args.phase == "sft":
            if not args.seed_data:
                raise ValueError("sft requires --seed-data")
            examples = [json.loads(line) for line in Path(args.seed_data).read_text().splitlines() if line.strip()]
        for update in range(args.updates):
            if args.phase == "sft":
                trainer.supervised_update(random.sample(examples, min(4, len(examples))))
                metrics = {"update": update, "phase": "sft"}
            else:
                rows = trainer.collect(env)
                loss = trainer.update(rows)
                metrics = {"update": update, "loss": loss, "transitions": len(rows),
                           "reward": sum(row["reward"] for row in rows),
                           **env.episode_metrics}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(metrics, flush=True)
        destination = output / "checkpoint"
        policy.save(destination)
        inference = Path(args.checkpoint) / "inference.py"
        if inference.is_file():
            shutil.copy2(inference, destination / "inference.py")
        torch.save({"critic": trainer.critic.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}, output / "training_state.pt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
