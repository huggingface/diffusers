from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, deprecate


@dataclass
class NewbiePipelineOutput(BaseOutput):
    images: List["PIL.Image.Image"]
    latents: Optional[torch.Tensor] = None


class NewbiePipeline(DiffusionPipeline):
    """
    NewBie image pipeline (NextDiT + Gemma3 + JinaCLIP + FLUX VAE).
    - Transformer: `NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP`
    - Scheduler: `FlowMatchEulerDiscreteScheduler`
    - VAE: FLUX-style `AutoencoderKL` with scale/shift
    - Text encoder: Gemma3 (from 🤗 Transformers)
    - CLIP encoder: JinaCLIPModel (from 🤗 Transformers, ``trust_remote_code=True``)
    """

    model_cpu_offload_seq = "text_encoder->clip_model->transformer->vae"

    def __init__(
        self,
        transformer,
        text_encoder,
        tokenizer,
        clip_model,
        clip_tokenizer,
        vae,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    ):
        super().__init__()

        if scheduler is None:
            scheduler = FlowMatchEulerDiscreteScheduler()

        self.register_modules(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            vae=vae,
            scheduler=scheduler,
        )

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def _get_vae_scale_shift(self) -> Tuple[float, float]:
        config = getattr(self.vae, "config", None)
        scale = getattr(config, "scaling_factor", None)
        shift = getattr(config, "shift_factor", None)

        if scale is None:
            scale = 0.3611
        if shift is None:
            shift = 0.1159

        return float(scale), float(shift)

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_h, latent_w = height // 8, width // 8
        shape = (batch_size, 16, latent_h, latent_w)

        if latents is not None:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}."
                )
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"Got a list of {len(generator)} generators, but batch_size={batch_size}."
                )
            latents = torch.stack(
                [
                    torch.randn(shape[1:], generator=g, device=device, dtype=dtype)
                    for g in generator
                ],
                dim=0,
            )
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @torch.no_grad()
    def _encode_prompt(
        self,
        prompts: List[str],
        clip_captions: Optional[List[str]] = None,
        max_length: int = 512,
        clip_max_length: int = 512,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if clip_captions is None:
            clip_captions = prompts

        # Gemma tokenizer + encoder
        text_inputs = self.tokenizer(
            prompts,
            padding=True,
            pad_to_multiple_of=8,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        attn_mask = text_inputs.attention_mask.to(self.text_encoder.device)

        enc_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        cap_feats = enc_out.hidden_states[-2]
        cap_mask = attn_mask

        # Jina CLIP encoding
        clip_inputs = self.clip_tokenizer(
            clip_captions,
            padding=True,
            truncation=True,
            max_length=clip_max_length,
            return_tensors="pt",
        ).to(self.clip_model.device)

        clip_feats = self.clip_model.get_text_features(input_ids=clip_inputs)

        clip_text_pooled: Optional[torch.Tensor] = None
        clip_text_sequence: Optional[torch.Tensor] = None

        if isinstance(clip_feats, (tuple, list)) and len(clip_feats) == 2:
            clip_text_pooled, clip_text_sequence = clip_feats
        else:
            clip_text_pooled = clip_feats

        if clip_text_sequence is not None:
            clip_text_sequence = clip_text_sequence.clone()
        if clip_text_pooled is not None:
            clip_text_pooled = clip_text_pooled.clone()

        clip_mask = clip_inputs.attention_mask

        return cap_feats, cap_mask, clip_text_sequence, clip_text_pooled, clip_mask

    # ---------------------------------------------------------------------
    # main call
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 5.0,
        cfg_trunc: float = 1.0,
        renorm_cfg: bool = True,
        system_prompt: str = "",
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        return_latents: bool = False,
        **kwargs,
    ) -> Union[NewbiePipelineOutput, Tuple[List["PIL.Image.Image"], torch.Tensor]]:


        if isinstance(prompt, str):
            batch_size = 1
            prompts = [prompt]
        else:
            prompts = list(prompt)
            batch_size = len(prompts)

        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            neg_prompts = [negative_prompt] * batch_size
        else:
            neg_prompts = list(negative_prompt)
            if len(neg_prompts) != batch_size:
                raise ValueError(
                    "negative_prompt must have same batch size as prompt when provided as a list."
                )

        if num_images_per_prompt != 1:
            deprecate(
                "num_images_per_prompt!=1 for NewbiePipeline",
                "0.31.0",
                "The Newbie architecture currently assumes num_images_per_prompt == 1.",
            )

        clip_captions_pos = prompts
        clip_captions_neg = neg_prompts

        if system_prompt:
            prompts_for_gemma = [system_prompt + p for p in prompts]
            neg_for_gemma = [system_prompt + p if p else "" for p in neg_prompts]
        else:
            prompts_for_gemma = prompts
            neg_for_gemma = neg_prompts

        device = self._execution_device
        dtype = self.transformer.dtype

        latents = self._prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        latents = latents.to(device=device, dtype=dtype)
        latents = latents.repeat(2, 1, 1, 1)  # [2B, C, H, W]

        full_gemma_prompts = prompts_for_gemma + neg_for_gemma
        full_clip_captions = clip_captions_pos + clip_captions_neg

        cap_feats, cap_mask, clip_text_sequence, clip_text_pooled, clip_mask = self._encode_prompt(
            full_gemma_prompts,
            clip_captions=full_clip_captions,
        )

        cap_feats = cap_feats.to(device=device, dtype=dtype)
        cap_mask = cap_mask.to(device)
        if clip_text_sequence is not None:
            clip_text_sequence = clip_text_sequence.to(device=device, dtype=dtype)
        if clip_text_pooled is not None:
            clip_text_pooled = clip_text_pooled.to(device=device, dtype=dtype)

        model_kwargs: Dict[str, Any] = dict(
            cap_feats=cap_feats,
            cap_mask=cap_mask,
            cfg_scale=float(guidance_scale),
            cfg_trunc=float(cfg_trunc),
            renorm_cfg=bool(renorm_cfg),
            clip_text_sequence=clip_text_sequence,
            clip_text_pooled=clip_text_pooled,
            clip_img_pooled=None,
        )

        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for t in timesteps:
            timestep = t

            noise_pred = self.transformer.forward_with_cfg(
                latents,
                timestep,
                **model_kwargs,
            )

            noise_pred = -noise_pred

            latents = self.scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=latents,
                return_dict=False,
            )[0]

        latents_out = latents[:batch_size]

        # 7. VAE decode
        vae_scale, vae_shift = self._get_vae_scale_shift()
        decoded = self.vae.decode(latents_out / vae_scale + vae_shift).sample
        images = (decoded / 2 + 0.5).clamp(0, 1)

        if output_type == "pil":
            import numpy as np
            from PIL import Image

            images_np = images.detach().float().cpu()
            images_np = images_np.permute(0, 2, 3, 1).numpy()
            images_np = (images_np * 255).round().astype(np.uint8)
            images_out = [Image.fromarray(img) for img in images_np]
        else:
            images_out = images

        if not return_dict:
            return images_out, (latents_out if return_latents else None)

        return NewbiePipelineOutput(
            images=images_out,
            latents=latents_out if return_latents else None,
        )


