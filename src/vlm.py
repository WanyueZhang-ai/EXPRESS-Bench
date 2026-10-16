import time
import logging
import torch
import numpy as np

from src.efficiency import current_profiler
from src.efficiency_models import model_call


class VLM:
    def __init__(self, cfg):
        from prismatic import load
        start_time = time.time()
        local_model_root = cfg.get("local_model_root") if hasattr(cfg, "get") else None
        self.model = current_profiler().measure("initialization.prismatic", load,
            cfg.model_id,
            hf_token=cfg.get("hf_token") if hasattr(cfg, "get") else cfg.hf_token,
            local_model_root=local_model_root,
            category="initialization",
        )
        self.model.to(cfg.device, dtype=torch.bfloat16)
        logging.info(f"Loaded VLM in {time.time() - start_time:.3f}s")

    def generate(self, prompt, image, T=0.4, max_tokens=512, *, purpose="generation"):
        efficiency = current_profiler()
        with model_call("prismatic", purpose, input_modality="text_image", output_kind="generation", device=self.model.device) as call:
            if call.enabled:
                call.layout = "inserted"
            with efficiency.span("input.prompt"):
                prompt_builder = self.model.get_prompt_builder()
                prompt_builder.add_turn(role="human", message=prompt)
                prompt_text = prompt_builder.get_prompt()
            with call.observe(self.model), efficiency.span("generation", device=self.model.device):
                return self.model.generate(image, prompt_text, do_sample=True, temperature=T,
                                           max_new_tokens=max_tokens, min_length=1, **call.streamer_kwargs())

    def get_loss(self, image, prompt, tokens, get_smx=True, T=1, *, purpose="candidate_scoring"):
        """Score candidate next tokens with one multimodal forward pass."""
        efficiency = current_profiler()
        with model_call("prismatic", purpose, input_modality="text_image", output_kind="scoring", device=self.model.device,
                        candidate_count=len(tokens), decision_fields=1) as call:
            with efficiency.span("input.prompt"):
                prompt_builder = self.model.get_prompt_builder()
                prompt_builder.add_turn(role="human", message=prompt)
                prompt_text = prompt_builder.get_prompt()
            tokenizer = self.model.llm_backbone.tokenizer
            with efficiency.span("input.tokenize"):
                input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids
            call.set_inputs({"input_ids": input_ids}, layout="inserted", source="prismatic_tokenizer_input_ids")
            with efficiency.span("input.image_preprocess"):
                pixel_values = self.model.vision_backbone.image_transform(image)
            with efficiency.span("input.to_device", device=self.model.device):
                input_ids = input_ids.to(self.model.device)
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.unsqueeze(0).to(self.model.device)
                elif isinstance(pixel_values, dict):
                    pixel_values = {key: value.unsqueeze(0).to(self.model.device) for key, value in pixel_values.items()}
                else:
                    raise TypeError(f"Unsupported transformed image type: {type(pixel_values)}")

            autocast_enabled = self.model.device.type == "cuda" and self.model.enable_mixed_precision_training
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=self.model.llm_backbone.half_precision_dtype, enabled=autocast_enabled
            ), call.observe(self.model):
                output = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                    pixel_values=pixel_values, use_cache=False, return_dict=True)
                with efficiency.span("output.score"):
                    token_ids = [self.model.string2idx[token] for token in tokens]
                    logits = output.logits[0, -1, token_ids].float()
                    if get_smx:
                        return torch.softmax(logits / T, dim=0).cpu().numpy()
                    return (-torch.log_softmax(logits, dim=0)).cpu().numpy()
