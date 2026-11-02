"""Differentiable decision-token scoring using the checkpoint's native encoder."""
import torch
from torch import nn
from torch.distributions import Categorical


class TrainableDecisionPolicy(nn.Module):
    def __init__(self, engine, temperature=1.0):
        super().__init__()
        self.backend = getattr(engine.model, "backend", None)
        if self.backend is None or not callable(getattr(self.backend, "encode", None)):
            raise ValueError("Training requires an HF backend exposing encode(row), model and tokenizer")
        self.network = self.backend.model
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def forward(self, request):
        compiled, batch, positions = self.backend.encode(request)
        device = next(self.network.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        # eval disables dropout while preserving gradients and PPO probability consistency.
        self.network.eval()
        logits = self.network(**batch, use_cache=False,
                              logits_to_keep=positions.to(device)).logits[0]
        distributions = {}
        labels = {}
        for index, field in enumerate(compiled.fields):
            spec = request["questions"][field]
            values = ["no", "yes"] if spec["type"] == "noul" else list(spec["criteria"])
            ids = [self.backend.tokenizer.encode(symbol, add_special_tokens=False)
                   for symbol in compiled.symbols[field]]
            if any(len(value) != 1 for value in ids):
                raise ValueError("decision symbols must each encode to one token")
            distributions[field] = Categorical(logits=logits[index, [value[0] for value in ids]].float() / self.temperature)
            labels[field] = values
        return distributions, labels

    def save(self, directory):
        self.network.save_pretrained(directory)
        self.backend.tokenizer.save_pretrained(directory)
        self.backend.processor.save_pretrained(directory)


class PrivilegedCritic(nn.Module):
    """Fixed-size visible-budget and training-only evidence/geometric features."""
    def __init__(self, input_size=6):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)
