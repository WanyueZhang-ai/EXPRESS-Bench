"""Local Qwen3-VL replacement for the former OpenAI GPT-4o calls."""

import os
from pathlib import Path
from typing import Optional, Sequence


QWEN3_VL_MODEL_PATH = os.environ.get(
    "QWEN3_VL_MODEL_PATH", "/ssd/wyzhang/all_models/Qwen3-VL-8B-Instruct"
)

_MODEL = None
_PROCESSOR = None
_MODEL_DEVICE = None


def _device_map_for(device):
    if device is None or not str(device).strip():
        return "auto"
    return {"": str(device)}


def prompt_make(prompt_path, ex_prompt):
    with open(prompt_path, "r", encoding="utf-8") as f:
        txt = f.readlines()
        prompt_system = txt[1]
        prompt = txt[3]
        if len(txt) > 4:
            for i in range(4, len(txt)):
                prompt = prompt + txt[i]
        prompt = prompt + ex_prompt
        return prompt_system, prompt


def _load_qwen3_vl(device=None):
    global _MODEL, _PROCESSOR, _MODEL_DEVICE
    requested_device = device if device is not None else os.environ.get("QWEN3_VL_DEVICE")
    if _MODEL is not None and _PROCESSOR is not None:
        if requested_device and _MODEL_DEVICE and str(requested_device) != str(_MODEL_DEVICE):
            raise RuntimeError(f"Qwen3-VL already loaded on {_MODEL_DEVICE}; requested {requested_device}")
        return _MODEL, _PROCESSOR

    model_path = Path(QWEN3_VL_MODEL_PATH).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen3-VL local model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Qwen3-VL directory is missing config.json: {model_path}")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    _MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype="auto",
        device_map=_device_map_for(requested_device),
        local_files_only=True,
    )
    _MODEL.eval()
    # PyTorch 2.2 used by Fine-EQA is incompatible with Transformers' fast
    # Qwen image processor; the slow processor has the same model inputs.
    _PROCESSOR = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    _MODEL_DEVICE = requested_device or "auto"
    return _MODEL, _PROCESSOR



def normalize_image_paths(
    img_path: str | Path | Sequence[str | Path], *, max_images: int = 4
) -> list[Path]:
    """Normalize one image or a bounded sequence while retaining input order."""
    if max_images < 1:
        raise ValueError("max_images must be positive")
    values = [img_path] if isinstance(img_path, (str, Path)) else list(img_path)
    if not values:
        raise ValueError("image count must be positive")
    paths = [Path(value).expanduser() for value in values[:max_images]]
    if any(not str(path) for path in paths):
        raise ValueError("image paths must be non-empty")
    return paths


def qwen3_vl(
    prompt_path: str,
    ex_prompt: str,
    img_path: Optional[str | Sequence[str]] = None,
    *,
    device: Optional[str] = None,
) -> str:
    """Generate a response with the local Qwen3-VL checkpoint.

    The function keeps the call signature used by ``main.py`` so region
    selection, exploration stopping, answer generation, and EAC scoring all
    use the same local model.
    """

    import torch

    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    user_content = []
    if img_path is not None:
        for image_path in normalize_image_paths(img_path):
            if not image_path.is_file():
                raise FileNotFoundError(f"Input image does not exist: {image_path}")
            user_content.append({"type": "image", "image": str(image_path)})
    user_content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt_system}]},
        {"role": "user", "content": user_content},
    ]

    model, processor = _load_qwen3_vl(device=device)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
