# Copyright (c) 2025, Baidu Inc. All rights reserved.
# Author: fengzhida (fengzhida@baidu.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ernie-Image Pipeline for HuggingFace Diffusers.
"""

import json
import os
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import BaseOutput
from ...models import AutoencoderKLFlux2
from ...models.transformers import ErnieImageTransformer2DModel


@dataclass
class ErnieImagePipelineOutput(BaseOutput):
    images: List[Image.Image]


class ErnieImagePipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using ErnieImageTransformer2DModel.

    This pipeline uses:
    - A custom DiT transformer model
    - A Flux2-style VAE for encoding/decoding latents
    - A text encoder (e.g., Qwen) for text conditioning
    - Flow Matching Euler Discrete Scheduler
    """

    model_cpu_offload_seq = "pe->text_encoder->transformer->vae"
    # For SGLang fallback ...
    _optional_components = ["pe", "pe_tokenizer"]
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer,
        vae,
        text_encoder,
        tokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        pe=None,
        pe_tokenizer=None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            pe=pe,
            pe_tokenizer=pe_tokenizer,
        )
        self.vae_scale_factor = 16  # VAE downsample factor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load pipeline from a pretrained model directory.

        Args:
            pretrained_model_name_or_path: Path to the saved pipeline directory
            **kwargs: Additional arguments passed to component loaders
                - torch_dtype: Data type for model weights (default: torch.bfloat16)
                - device_map: Device map for model loading
                - trust_remote_code: Whether to trust remote code for text encoder

        Returns:
            ErnieImagePipeline instance
        """

        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        device_map = kwargs.pop("device_map", None)

        # Determine whether this is a local directory or a Hub repo ID.
        # For local paths we join sub-directories; for Hub IDs we use `subfolder`.
        is_local = os.path.isdir(pretrained_model_name_or_path)

        def _path_or_subfolder(subfolder: str):
            if is_local:
                return {"pretrained_model_name_or_path": os.path.join(pretrained_model_name_or_path, subfolder)}
            return {"pretrained_model_name_or_path": pretrained_model_name_or_path, "subfolder": subfolder}

        # Load transformer
        transformer = ErnieImageTransformer2DModel.from_pretrained(
            **_path_or_subfolder("transformer"),
            torch_dtype=torch_dtype,
        )

        # Load VAE
        vae = AutoencoderKLFlux2.from_pretrained(
            **_path_or_subfolder("vae"),
            torch_dtype=torch_dtype,
        )

        # Load text encoder
        text_encoder = AutoModel.from_pretrained(
            **_path_or_subfolder("text_encoder"),
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            **_path_or_subfolder("tokenizer"),
            trust_remote_code=trust_remote_code,
        )

        # Load PE
        pe = AutoModelForCausalLM.from_pretrained(
            **_path_or_subfolder("pe"),
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            **({"device_map": device_map} if device_map else {}),
        )

        # Load PE tokenizer (auto-picks up chat_template.jinja in the same dir)
        pe_tokenizer = AutoTokenizer.from_pretrained(
            **_path_or_subfolder("pe"),
            trust_remote_code=trust_remote_code,
        )

        # Load scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            **_path_or_subfolder("scheduler"),
        )

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            pe=pe,
            pe_tokenizer=pe_tokenizer,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def _enhance_prompt_with_pe(
        self,
        prompt: str,
        device: torch.device,
        width: int = 1024,
        height: int = 1024,
        system_prompt: Optional[str] = None,
        max_length: int = 1536,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        """Use PE model to rewrite/enhance a short prompt via chat_template."""
        # Build user message as JSON carrying prompt text and target resolution
        user_content = json.dumps(
            {"prompt": prompt, "width": width, "height": height},
            ensure_ascii=False,
        )
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # apply_chat_template picks up the chat_template.jinja loaded with pe_tokenizer
        input_text = self.pe_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # "Output:" is already in the user block
        )
        # When accelerate offload hooks are installed, use the hook's execution_device
        # to ensure inputs land on the same device as the model weights during forward()
        if hasattr(self.pe, "_hf_hook") and hasattr(self.pe._hf_hook, "execution_device"):
            pe_device = self.pe._hf_hook.execution_device
        else:
            pe_device = device
        inputs = self.pe_tokenizer(input_text, return_tensors="pt").to(pe_device)

        output_ids = self.pe.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=temperature != 1.0 or top_p != 1.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.pe_tokenizer.pad_token_id,
            eos_token_id=self.pe_tokenizer.eos_token_id,
        )
        # Decode only newly generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.pe_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        max_length: int = 64,
    ) -> List[torch.Tensor]:
        """Encode text prompts to embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]

        text_hiddens = []

        for p in prompt:
            ids = self.tokenizer(
                p,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )["input_ids"]

            if len(ids) == 0:
                if self.tokenizer.bos_token_id is not None:
                    ids = [self.tokenizer.bos_token_id]
                else:
                    ids = [0]

            input_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )
                # Use second to last hidden state (matches training)
                hidden = outputs.hidden_states[-2][0]  # [T, H]

            # Repeat for num_images_per_prompt
            for _ in range(num_images_per_prompt):
                text_hiddens.append(hidden)

        return text_hiddens

    @torch.no_grad()
    def _encode_negative_prompt(
        self,
        negative_prompt: List[str],
        device: torch.device,
        num_images_per_prompt: int = 1,
        max_length: int = 64,
    ) -> List[torch.Tensor]:
        """Encode negative prompts for CFG."""
        text_hiddens = []

        for np in negative_prompt:
            ids = self.tokenizer(
                np,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )["input_ids"]

            if len(ids) == 0:
                if self.tokenizer.bos_token_id is not None:
                    ids = [self.tokenizer.bos_token_id]
                else:
                    ids = [0]

            input_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-2][0]

            for _ in range(num_images_per_prompt):
                text_hiddens.append(hidden)

        return text_hiddens

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """2x2 patchify: [B, 32, H, W] -> [B, 128, H/2, W/2]"""
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Reverse patchify: [B, 128, H/2, W/2] -> [B, 32, H, W]"""
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_length: int = 1536,
        use_pe: bool = True,    # 默认使用PE进行改写
    ):
        """
        Generate images from text prompts.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG. Default is "".
            height: Image height (must be divisible by 16)
            width: Image width (must be divisible by 16)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (1.0 = no guidance)
            num_images_per_prompt: Number of images per prompt
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            output_type: "pil" or "latent"
            return_dict: Whether to return a dataclass
            callback_on_step_end: Optional callback invoked at the end of each denoising step.
                Called as `callback_on_step_end(pipeline, step, timestep, callback_kwargs)` where
                `callback_kwargs` contains the tensors listed in `callback_on_step_end_tensor_inputs`.
                The callback may return a dict to override those tensors for subsequent steps.
            callback_on_step_end_tensor_inputs: List of tensor names passed into the callback kwargs.
                Must be a subset of `_callback_tensor_inputs` (default: `["latents"]`).
            max_length: Max token length for text encoding

        Returns:
            Generated images
        """
        device = self._execution_device
        dtype = self.transformer.dtype

        # Validate dimensions
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"Height and width must be divisible by {self.vae_scale_factor}")

        # Handle prompts
        if isinstance(prompt, str):
            prompt = [prompt]

        # [Phase 1] PE: enhance prompts
        if use_pe and self.pe is not None and self.pe_tokenizer is not None:
            prompt = [
                self._enhance_prompt_with_pe(p, device, width=width, height=height, max_length=max_length)
                for p in prompt
            ]

        batch_size = len(prompt)
        total_batch_size = batch_size * num_images_per_prompt

        # Handle negative prompt
        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if len(negative_prompt) != batch_size:
            raise ValueError(f"negative_prompt must have same length as prompt ({batch_size})")

        # [Phase 2] Text encoding
        text_hiddens = self.encode_prompt(prompt, device, num_images_per_prompt, max_length)

        # CFG with negative prompt
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            uncond_text_hiddens = self._encode_negative_prompt(
                negative_prompt, device, num_images_per_prompt, max_length
            )

        # Latent dimensions
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = 128  # After patchify

        # Initialize latents
        if latents is None:
            latents = torch.randn(
                (total_batch_size, latent_channels, latent_h, latent_w),
                device=device,
                dtype=dtype,
                generator=generator,
            )

        # Setup scheduler
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)

        # Denoising loop
        if do_cfg:
            cfg_text_hiddens = list(uncond_text_hiddens) + list(text_hiddens)
        else:
            cfg_text_hiddens = text_hiddens
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                if do_cfg:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    t_batch = torch.full((total_batch_size * 2,), t.item(), device=device, dtype=dtype)
                else:
                    latent_model_input = latents
                    t_batch = torch.full((total_batch_size,), t.item(), device=device, dtype=dtype)

                # Model prediction
                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    encoder_hidden_states=cfg_text_hiddens,
                    return_dict=False,
                )[0]

                # Apply CFG
                if do_cfg:
                    pred_uncond, pred_cond = pred.chunk(2, dim=0)
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # Scheduler step
                latents = self.scheduler.step(pred, t, latents).prev_sample

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        if output_type == "latent":
            return latents

        # Decode latents to images
        # Unnormalize latents using VAE's BN stats
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(device)
        latents = latents * bn_std + bn_mean

        # Unpatchify
        latents = self._unpatchify_latents(latents)

        # Decode
        images = self.vae.decode(latents, return_dict=False)[0]

        # Post-process
        images = (images.clamp(-1, 1) + 1) / 2
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)

        return ErnieImagePipelineOutput(images=images)
