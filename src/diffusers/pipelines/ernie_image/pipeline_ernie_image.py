# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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
from typing import Callable, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ...models import AutoencoderKLFlux2
from ...models.transformers import ErnieImageTransformer2DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from .pipeline_output import ErnieImagePipelineOutput


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
        transformer: ErnieImageTransformer2DModel,
        vae: AutoencoderKLFlux2,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        pe: Optional[AutoModelForCausalLM] = None,
        pe_tokenizer: Optional[AutoTokenizer] = None,
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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels)) if getattr(self, "vae", None) else 16

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @torch.no_grad()
    def _enhance_prompt_with_pe(
        self,
        prompt: str,
        device: torch.device,
        width: int = 1024,
        height: int = 1024,
        system_prompt: Optional[str] = None,
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
        inputs = self.pe_tokenizer(input_text, return_tensors="pt").to(device)
        output_ids = self.pe.generate(
            **inputs,
            max_new_tokens=self.pe_tokenizer.model_max_length,
            do_sample=temperature != 1.0 or top_p != 1.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.pe_tokenizer.pad_token_id,
            eos_token_id=self.pe_tokenizer.eos_token_id,
        )
        # Decode only newly generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.pe_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
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

    @staticmethod
    def _pad_text(text_hiddens: List[torch.Tensor], device: torch.device, dtype: torch.dtype, text_in_dim: int):
        B = len(text_hiddens)
        if B == 0:
            return torch.zeros((0, 0, text_in_dim), device=device, dtype=dtype), torch.zeros(
                (0,), device=device, dtype=torch.long
            )
        normalized = [
            th.squeeze(1).to(device).to(dtype) if th.dim() == 3 else th.to(device).to(dtype) for th in text_hiddens
        ]
        lens = torch.tensor([t.shape[0] for t in normalized], device=device, dtype=torch.long)
        Tmax = int(lens.max().item())
        text_bth = torch.zeros((B, Tmax, text_in_dim), device=device, dtype=dtype)
        for i, t in enumerate(normalized):
            text_bth[i, : t.shape[0], :] = t
        return text_bth, lens

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: list[torch.FloatTensor] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_pe: bool = True,  # 默认使用PE进行改写
    ):
        """
        Generate images from text prompts.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG. Default is "".
            height: Image height in pixels (must be divisible by 16). Default: 1024.
            width: Image width in pixels (must be divisible by 16). Default: 1024.
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (1.0 = no guidance). Default: 4.0.
            num_images_per_prompt: Number of images per prompt
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            prompt_embeds: Pre-computed text embeddings for positive prompts (optional).
                If provided, `encode_prompt` is skipped for positive prompts.
            negative_prompt_embeds: Pre-computed text embeddings for negative prompts (optional).
                If provided, `encode_prompt` is skipped for negative prompts.
            output_type: "pil" or "latent"
            return_dict: Whether to return a dataclass
            callback_on_step_end: Optional callback invoked at the end of each denoising step.
                Called as `callback_on_step_end(pipeline, step, timestep, callback_kwargs)` where `callback_kwargs`
                contains the tensors listed in `callback_on_step_end_tensor_inputs`. The callback may return a dict to
                override those tensors for subsequent steps.
            callback_on_step_end_tensor_inputs: List of tensor names passed into the callback kwargs.
                Must be a subset of `_callback_tensor_inputs` (default: `["latents"]`).
            use_pe: Whether to use the PE model to enhance prompts before generation.

        Returns:
            :class:`ErnieImagePipelineOutput` with `images` and `revised_prompts`.
        """
        device = self._execution_device
        dtype = self.transformer.dtype

        self._guidance_scale = guidance_scale

        # Validate prompt / prompt_embeds
        if prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt` and `prompt_embeds` at the same time.")

        # Validate dimensions
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"Height and width must be divisible by {self.vae_scale_factor}")

        # Handle prompts
        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]

        # [Phase 1] PE: enhance prompts
        revised_prompts: Optional[List[str]] = None
        if prompt is not None and use_pe and self.pe is not None and self.pe_tokenizer is not None:
            prompt = [self._enhance_prompt_with_pe(p, device, width=width, height=height) for p in prompt]
            revised_prompts = list(prompt)

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)
        total_batch_size = batch_size * num_images_per_prompt

        # Handle negative prompt
        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if len(negative_prompt) != batch_size:
            raise ValueError(f"negative_prompt must have same length as prompt ({batch_size})")

        # [Phase 2] Text encoding
        if prompt_embeds is not None:
            text_hiddens = prompt_embeds
        else:
            text_hiddens = self.encode_prompt(prompt, device, num_images_per_prompt)

        # CFG with negative prompt
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                uncond_text_hiddens = negative_prompt_embeds
            else:
                uncond_text_hiddens = self.encode_prompt(negative_prompt, device, num_images_per_prompt)

        # Latent dimensions
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels  # After patchify

        # Initialize latents
        if latents is None:
            latents = randn_tensor(
                (total_batch_size, latent_channels, latent_h, latent_w),
                generator=generator,
                device=device,
                dtype=dtype,
            )

        # Setup scheduler
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)

        # Denoising loop
        if self.do_classifier_free_guidance:
            cfg_text_hiddens = list(uncond_text_hiddens) + list(text_hiddens)
        else:
            cfg_text_hiddens = text_hiddens
        text_bth, text_lens = self._pad_text(
            text_hiddens=cfg_text_hiddens, device=device, dtype=dtype, text_in_dim=self.transformer.config.text_in_dim
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    t_batch = torch.full((total_batch_size * 2,), t.item(), device=device, dtype=dtype)
                else:
                    latent_model_input = latents
                    t_batch = torch.full((total_batch_size,), t.item(), device=device, dtype=dtype)

                # Model prediction
                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    return_dict=False,
                )[0]

                # Apply CFG
                if self.do_classifier_free_guidance:
                    pred_uncond, pred_cond = pred.chunk(2, dim=0)
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # Scheduler step
                latents = self.scheduler.step(pred, t, latents).prev_sample

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
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

        return ErnieImagePipelineOutput(images=images, revised_prompts=revised_prompts)
