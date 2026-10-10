from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.vlm import VLM


class _PromptBuilder:
    def add_turn(self, role, message):
        self.message = message

    def get_prompt(self):
        return self.message


class _Tokenizer:
    def __call__(self, text, truncation, return_tensors):
        assert text == "prompt"
        assert truncation and return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))


class _FakeModel:
    device = torch.device("cpu")
    enable_mixed_precision_training = True
    string2idx = {"Yes": 1, "No": 2}
    llm_backbone = SimpleNamespace(tokenizer=_Tokenizer(), half_precision_dtype=torch.bfloat16)

    def __init__(self, image_transform):
        self.vision_backbone = SimpleNamespace(image_transform=image_transform)

    def get_prompt_builder(self):
        return _PromptBuilder()

    def __call__(self, *, input_ids, attention_mask, pixel_values, use_cache, return_dict):
        assert input_ids.tolist() == [[1, 2]]
        assert attention_mask.tolist() == [[1, 1]]
        assert use_cache is False
        assert return_dict is True
        if isinstance(pixel_values, dict):
            assert set(pixel_values) == {"dino", "siglip"}
            assert all(value.shape == (1, 3, 2, 2) for value in pixel_values.values())
        else:
            assert pixel_values.shape == (1, 3, 2, 2)
        logits = torch.tensor([[[0.0, 0.0, 0.0], [0.0, np.log(0.8), np.log(0.2)]]])
        return SimpleNamespace(logits=logits)


def _vlm(image_transform):
    vlm = VLM.__new__(VLM)
    vlm.model = _FakeModel(image_transform)
    return vlm


def test_get_loss_scores_next_token_without_generation():
    vlm = _vlm(lambda image: torch.ones(3, 2, 2))
    result = vlm.get_loss(Image.new("RGB", (2, 2)), "prompt", ["Yes", "No"])
    np.testing.assert_allclose(result, [0.8, 0.2], rtol=1e-6)
    losses = vlm.get_loss(Image.new("RGB", (2, 2)), "prompt", ["Yes", "No"], get_smx=False)
    np.testing.assert_allclose(losses, -np.log([0.8, 0.2]), rtol=1e-6)


def test_get_loss_supports_dinosiglip_image_transform():
    vlm = _vlm(lambda image: {"dino": torch.ones(3, 2, 2), "siglip": torch.ones(3, 2, 2)})
    result = vlm.get_loss(Image.new("RGB", (2, 2)), "prompt", ["Yes", "No"], T=2)
    expected = np.sqrt([0.8, 0.2])
    np.testing.assert_allclose(result, expected / expected.sum(), rtol=1e-6)
