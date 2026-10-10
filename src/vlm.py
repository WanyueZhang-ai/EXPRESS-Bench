import time
import logging
import torch
import numpy as np

from prismatic import load


class VLM:
    def __init__(self, cfg):
        start_time = time.time()
        local_model_root = cfg.get("local_model_root") if hasattr(cfg, "get") else None
        self.model = load(
            cfg.model_id,
            hf_token=cfg.get("hf_token") if hasattr(cfg, "get") else cfg.hf_token,
            local_model_root=local_model_root,
        )
        self.model.to(cfg.device, dtype=torch.bfloat16)
        logging.info(f"Loaded VLM in {time.time() - start_time:.3f}s")

    def generate(self, prompt, image, T=0.4, max_tokens=512):
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        generated_text = self.model.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=T,
            max_new_tokens=max_tokens,
            min_length=1,
        )
        return generated_text

    def get_loss(self, image, prompt, tokens, get_smx=True, T=1):
        """Score candidate next tokens with one multimodal forward pass."""
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()

        tokenizer = self.model.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.model.device)
        pixel_values = self.model.vision_backbone.image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.unsqueeze(0).to(self.model.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {key: value.unsqueeze(0).to(self.model.device) for key, value in pixel_values.items()}
        else:
            raise TypeError(f"Unsupported transformed image type: {type(pixel_values)}")

        autocast_enabled = self.model.device.type == "cuda" and self.model.enable_mixed_precision_training
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=self.model.llm_backbone.half_precision_dtype, enabled=autocast_enabled
        ):
            output = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=pixel_values,
                use_cache=False,
                return_dict=True,
            )
            token_ids = [self.model.string2idx[token] for token in tokens]
            logits = output.logits[0, -1, token_ids].float()
            if get_smx:
                return torch.softmax(logits / T, dim=0).cpu().numpy()
            return (-torch.log_softmax(logits, dim=0)).cpu().numpy()
