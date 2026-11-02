"""On-policy PPO, auxiliary supervision, and supervised seed initialization."""
import copy
import torch


def gae(rewards, values, next_values, terminated, boundaries, gamma=0.99, lam=0.95):
    advantages = [0.0] * len(rewards)
    carry = 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * next_values[index] * (not terminated[index]) - values[index]
        carry = delta + gamma * lam * (not boundaries[index]) * carry
        advantages[index] = carry
    return advantages, [adv + value for adv, value in zip(advantages, values)]


def auxiliary_loss(distributions, labels, targets):
    loss = distributions["action"].logits.sum() * 0
    for field, target in targets.items():
        if field not in distributions:
            continue
        options = labels[field]
        label = str(target)
        if label not in options:
            raise ValueError(f"invalid supervised label {field}={label}")
        index = torch.tensor(options.index(label), device=distributions[field].logits.device)
        loss = loss - distributions[field].log_prob(index)
    return loss


class PPOTrainer:
    def __init__(self, policy, critic, learning_rate=1e-5, clip=0.2, entropy=0.01):
        self.policy, self.critic = policy, critic
        self.clip, self.entropy = clip, entropy
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad] + list(critic.parameters()),
            lr=learning_rate,
        )
        self.device = next(policy.parameters()).device
        critic.to(self.device)

    def value(self, features):
        return self.critic(torch.tensor(features, dtype=torch.float32, device=self.device))

    def collect(self, env, episodes=1):
        rows = []
        for _ in range(episodes):
            request, features = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    distributions, labels = self.policy(request)
                    action_index = distributions["action"].sample()
                    action = labels["action"][action_index.item()]
                    old_log_probability = distributions["action"].log_prob(action_index).item()
                    value = self.value(features).item()
                branch_targets = env.counterfactual_labels()
                next_request, next_features, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                with torch.no_grad():
                    next_value = 0.0 if terminated else self.value(next_features).item()
                rows.append({"request": copy.deepcopy(request), "features": list(features),
                             "action": action, "old_log_probability": old_log_probability,
                             "value": value, "next_value": next_value, "reward": reward,
                             "terminated": terminated, "boundary": done,
                             "targets": dict(branch_targets, **info.get("targets", {}))})
                request, features = next_request, next_features
        advantages, returns = gae(
            [row["reward"] for row in rows], [row["value"] for row in rows],
            [row["next_value"] for row in rows], [row["terminated"] for row in rows],
            [row["boundary"] for row in rows],
        )
        for row, advantage, target_return in zip(rows, advantages, returns):
            row.update(advantage=advantage, target_return=target_return)
        return rows

    def update(self, rows, epochs=4, auxiliary_weight=0.1):
        if not rows:
            raise ValueError("empty on-policy rollout")
        advantages = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        losses = []
        for _ in range(epochs):
            self.optimizer.zero_grad()
            for row, advantage in zip(rows, advantages):
                distributions, labels = self.policy(row["request"])
                distribution = distributions["action"]
                action_index = torch.tensor(labels["action"].index(row["action"]), device=self.device)
                ratio = (distribution.log_prob(action_index) - row["old_log_probability"]).exp()
                advantage = advantage.to(self.device)
                policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1-self.clip, 1+self.clip) * advantage)
                value_loss = (self.value(row["features"]) - row["target_return"]) ** 2
                loss = policy_loss + 0.5 * value_loss - self.entropy * distribution.entropy()
                loss = loss + auxiliary_weight * auxiliary_loss(distributions, labels, row["targets"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss")
                (loss / len(rows)).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.critic.parameters()), 1.0)
            self.optimizer.step()
        return sum(losses) / len(losses)

    def supervised_update(self, examples):
        if not examples:
            raise ValueError("empty supervised batch")
        self.optimizer.zero_grad()
        for example in examples:
            distributions, labels = self.policy(example["request"])
            loss = auxiliary_loss(distributions, labels, example["targets"])
            (loss / len(examples)).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
