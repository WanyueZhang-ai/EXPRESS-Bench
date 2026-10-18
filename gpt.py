"""Local Qwen3-VL replacement for the former OpenAI GPT-4o calls."""

import os
from pathlib import Path
from typing import Optional
from src.efficiency import current_profiler
from src.efficiency_models import model_call


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


def qwen3_vl(
    prompt_path: str,
    ex_prompt: str,
    img_path: Optional[str] = None,
    *,
    device: Optional[str] = None,
) -> str:
    """Generate a response with the local Qwen3-VL checkpoint.

    The function keeps the call signature used by ``main.py`` so region
    selection, exploration stopping, answer generation, and EAC scoring all
    use the same local model.
    """

    import torch

    efficiency = current_profiler()
    if _MODEL is None:
        model, processor = efficiency.measure("initialization.qwen3_vl", _load_qwen3_vl, device=device, category="initialization")
    else:
        model, processor = _load_qwen3_vl(device=device)
    purpose = {"region": "region_selection", "explore": "stop_check", "answer": "final_answer",
               "random_answer": "fallback_answer", "evaluation": "eac_evaluation"}.get(Path(prompt_path).stem, "generation")
    with model_call("qwen3_vl", purpose, input_modality="text_image" if img_path is not None else "text",
                    output_kind="generation", device=model.device, checkpoint=QWEN3_VL_MODEL_PATH,
                    image_count=int(img_path is not None), max_new_tokens=512, do_sample=False) as call:
        return _qwen_generate(model, processor, prompt_path, ex_prompt, img_path, call, efficiency)


def _qwen_generate(model, processor, prompt_path, ex_prompt, img_path, call, efficiency):
    import torch

    prompt_system, prompt = efficiency.measure("input.prompt", prompt_make, prompt_path, ex_prompt)
    user_content = []
    if img_path is not None:
        image_path = Path(img_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")
        user_content.append({"type": "image", "image": str(image_path)})
    user_content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt_system}]},
        {"role": "user", "content": user_content},
    ]

    with efficiency.span("input.processor"):
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
    visual_id = getattr(model.config, "image_token_id", None)
    call.set_inputs(inputs, visual_token_ids={visual_id} if isinstance(visual_id, int) else None,
                    source="processor_expanded_input_ids")
    with efficiency.span("input.to_device", device=model.device):
        inputs = inputs.to(model.device)

    with torch.inference_mode(), call.observe(model), efficiency.span("generation", device=model.device):
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            **call.streamer_kwargs(),
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    call.set_generated(generated_ids_trimmed)
    with efficiency.span("output.decode"):
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
