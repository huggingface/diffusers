# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ...models.cross_attention import LoRACrossAttnProcessor
from ...models.attention import BasicTransformerBlock
from ..stable_diffusion import StableDiffusionPipeline
from ...schedulers import EulerAncestralDiscreteScheduler
from . import FabricPipelineOutput

from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def apply_unet_lora_weights(pipeline, unet_path):
    model_weight = torch.load(unet_path, map_location="cpu")
    unet = pipeline.unet
    lora_attn_procs = {}
    lora_rank = list(
        set([v.size(0) for k, v in model_weight.items() if k.endswith("down.weight")])
    )
    assert len(lora_rank) == 1
    lora_rank = lora_rank[0]
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=lora_rank,
        ).to(pipeline.device)
    unet.set_attn_processor(lora_attn_procs)
    unet.load_state_dict(model_weight, strict=False)


def attn_with_weights(
    attn: nn.Module,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    weights=None,  # shape: (batch_size, sequence_length)
    lora_scale=1.0,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    if isinstance(attn.processor, LoRACrossAttnProcessor):
        query = attn.to_q(hidden_states) + lora_scale * attn.processor.to_q_lora(
            hidden_states
        )
    else:
        query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    if isinstance(attn.processor, LoRACrossAttnProcessor):
        key = attn.to_k(encoder_hidden_states) + lora_scale * attn.processor.to_k_lora(
            encoder_hidden_states
        )
        value = attn.to_v(
            encoder_hidden_states
        ) + lora_scale * attn.processor.to_v_lora(encoder_hidden_states)
    else:
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)

    if weights is not None:
        if weights.shape[0] != 1:
            weights = weights.repeat_interleave(attn.heads, dim=0)
        attention_probs = attention_probs * weights[:, None]
        attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    if isinstance(attn.processor, LoRACrossAttnProcessor):
        hidden_states = attn.to_out[0](
            hidden_states
        ) + lora_scale * attn.processor.to_out_lora(hidden_states)
    else:
        hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


class FabricPipeline(DiffusionPipeline):
    def __init__(
        self,
        model_name: Optional[str] = None,
        stable_diffusion_version: str = "1.5",
        scheduler: EulerAncestralDiscreteScheduler = EulerAncestralDiscreteScheduler,
        lora_weights: Optional[str] = None,
        torch_dtype = None,
    ):
        super().__init__()

        if stable_diffusion_version == "2.1":
            warnings.warn("StableDiffusion v2.x is not supported and may give unexpected results.")

        if model_name is None:
            if stable_diffusion_version == "1.5":
                model_name = "runwayml/stable-diffusion-v1-5"
            elif stable_diffusion_version == "2.1":
                model_name = "stabilityai/stable-diffusion-2-1"
            else:
                raise ValueError(
                    f"Unknown stable diffusion version: {stable_diffusion_version}. Version must be either '1.5' or '2.1'"
                )

        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
            safety_checker=None,
        ).to("cuda")

        if lora_weights:
            print(f"Applying LoRA weights from {lora_weights}")
            apply_unet_lora_weights(
                pipeline=pipe, unet_path=lora_weights
            )

        self.pipeline = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = scheduler
        
        self.dtype = torch_dtype

    #@property
    #def device(self):
    #    return next(self.parameters()).device

    #def to(self, device):
    #    self.pipeline.to(device)
    #    return super().to(device)

    def initialize_prompts(self, prompts: List[str], device):
        # Breaking into individual prompts feels memory efficient 
        prompt_embed_list = []
        for prompt in prompts:
          prompt_tokens = self.tokenizer(
              prompt,
              return_tensors="pt",
              max_length=self.tokenizer.model_max_length,
              padding="max_length",
              truncation=True,
          )

          attention_mask = prompt_tokens.attention_mask.to(device) if (
              hasattr(self.text_encoder.config, "use_attention_mask")
              and self.text_encoder.config.use_attention_mask
          ) else None

          prompt_embd = self.text_encoder(
              input_ids=prompt_tokens.input_ids.to(device),
              attention_mask=attention_mask,
          ).last_hidden_state
          
          prompt_embed_list.append(prompt_embd)

        return torch.cat(prompt_embed_list, dim=0)

    def get_unet_hidden_states(self, z_all, t, prompt_embd):
        cached_hidden_states = []
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):

                def new_forward(self, hidden_states, *args, **kwargs):
                    cached_hidden_states.append(hidden_states.clone().detach().cpu())
                    return self.old_forward(hidden_states, *args, **kwargs)

                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)

        # run forward pass to cache hidden states, output can be discarded
        _ = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return cached_hidden_states

    def unet_forward_with_cached_hidden_states(
        self,
        z_all,
        t,
        prompt_embd,
        cached_pos_hiddens: Optional[List[torch.Tensor]] = None,
        cached_neg_hiddens: Optional[List[torch.Tensor]] = None,
        pos_weights=(0.8, 0.8),
        neg_weights=(0.5, 0.5),
    ):
        if cached_pos_hiddens is None and cached_neg_hiddens is None:
            return self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        local_pos_weights = torch.linspace(
            *pos_weights, steps=len(self.unet.down_blocks) + 1)[:-1]
        local_neg_weights = torch.linspace(
            *neg_weights, steps=len(self.unet.down_blocks) + 1)[:-1]

        def new_forward_caching(module, hidden_states, cached_hiddens, weight, is_positive):
            cached_hs = cached_hiddens.pop(0).to(
                hidden_states.device
            )
            cond_hs = torch.cat(
                [hidden_states, cached_hs], dim=1
            )
            weights = weights.clone().repeat(
                1, 1 + cached_pos_hs.shape[1] // d_model
            )
            weights = torch.full((cond_hs.size(0), cond_hs.size(1) // hidden_states.size(1)), 
                weight, device=hidden_states.device)
            weights[:, hidden_states.size(1):] = 1.0
            out = attn_with_weights(
                self,
                hidden_states,
                encoder_hidden_states=cond_hs,
                weights=weights,
            )
            return out


        for block, pos_weight, neg_weight in zip(
            self.unet.down_blocks + [self.unet.mid_block] + self.unet.up_blocks,
            local_pos_weights + [pos_weights[1]] + local_pos_weights[::-1],
            local_neg_weights + [neg_weights[1]] + local_neg_weights[::-1],
        ):
            for module in block.modules():
                if isinstance(module, BasicTransformerBlock):

                    def new_forward(
                        self,
                        hidden_states,
                        pos_weight=pos_weight,
                        neg_weight=neg_weight,
                        **kwargs,
                    ):
                        cond_hiddens, uncond_hiddens = hidden_states.chunk(2, dim=0)
                        batch_size, d_model = cond_hiddens.shape[:2]
                        device, dtype = hidden_states.device, hidden_states.dtype

                        weights = torch.ones(
                            batch_size, d_model, device=device, dtype=dtype
                        )

                        out_pos = self.old_forward(hidden_states)
                        out_neg = self.old_forward(hidden_states)

                        if cached_pos_hiddens is not None:
                            out_pos = new_forward_caching(
                                self, hidden_states, cached_pos_hiddens, 
                                pos_weight, is_positive=True)


                        if cached_neg_hiddens is not None:
                            out_neg = new_forward_caching(
                                self, hidden_states, cached_neg_hiddens, 
                                neg_weight, is_positive=False)

                        out = torch.cat([out_pos, out_neg], dim=0)
                        return out

                    module.attn1.old_forward = module.attn1.forward
                    module.attn1.forward = new_forward.__get__(module.attn1)

        out = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return out

    def preprocess_feedback_images(images, vae, device) -> torch.tensor:
        images_t = [self.image_to_tensor(img) for img in images]
        images_t = torch.stack(images_t).to(device, dtype=self.dtype)
        latents = (
            vae.config.scaling_factor
            * vae.encode(iamges_t).latent_dist.sample()
        )
        return latents

    @torch.no_grad()

    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt: Optional[Union[str, List[str]]] = "",
        liked: Optional[List[Image.Image]] = [],
        disliked: Optional[List[Image.Image]] = [],
        random_seed: int = 42,
        n_images: int = 1,
        guidance_scale: float = 8.0,
        denoising_steps: int = 20,
        feedback_start_ratio: float = 0.33,
        feedback_end_ratio: float = 0.66,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
        neg_scale: float = 0.5,
        pos_bottleneck_scale: float = 1.0,
        neg_bottleneck_scale: float = 1.0,
    ):
        """
        Generate a trajectory of images with binary feedback.
        The feedback can be given as a list of liked and disliked images.
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        device = torch.device("cuda")

        latent_noise = torch.randn(n_images, 4, 64, 64, device=device, dtype=self.dtype)

        positive_latents = self.preprocess_feedback_images(liked,self.vae,device) if liked and len(liked)>1 else torch.tensor([], device=device, dtype=self.dtype)

        negative_latents =  self.preprocess_feedback_images(disliked,self.vae,device) if disliked and len(disliked)>0 else torch.tensor([], device=device, dtype=self.dtype)

        if isinstance(prompt, str):
            prompt = [prompt] * n_images
        else:
            assert len(prompt) == n_images
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * n_images
        else:
            assert len(negative_prompt) == n_images

        
        (cond_prompt_embs, uncond_prompt_embs, null_prompt_emb) = self.initialize_prompts(prompt + negative_prompt + [""], device).split([n_images, n_images, 1])

        batched_prompt_embd = torch.cat([cond_prompt_embs, uncond_prompt_embs], dim=0)

        self.scheduler.set_timesteps(denoising_steps, device=device)
        timesteps = self.scheduler.timesteps

        latent_noise = latent_noise * self.scheduler.init_noise_sigma

        num_warmup_steps = len(timesteps) - denoising_steps * self.scheduler.order

        ref_start_idx = round(len(timesteps) * feedback_start_ratio)
        ref_end_idx = round(len(timesteps) * feedback_end_ratio)

        with tqdm(total=denoising_steps) as pbar:
            for i, t in enumerate(timesteps):
                sigma = self.scheduler.sigma_t[t] if hasattr(self.scheduler, 'sigma_t') else 0
                if hasattr(self.scheduler, "sigmas"):
                    sigma = self.scheduler.sigmas[i]

                alpha_hat = 1 / (sigma**2 + 1)

                z_single = self.scheduler.scale_model_input(latent_noise, t)
                z_all = torch.cat([z_single] * 2, dim=0)
                z_ref = torch.cat([positive_latents, negative_latents], dim=0)

                if i >= ref_start_idx and i <= ref_end_idx:
                    weight_factor = max_weight
                else:
                    weight_factor = min_weight

                pos_ws = (weight_factor, weight_factor * pos_bottleneck_scale)
                neg_ws = (weight_factor * neg_scale, weight_factor * neg_scale * neg_bottleneck_scale)

                if z_ref.size(0) > 0 and weight_factor > 0:
                    noise = torch.randn_like(z_ref)
                    if isinstance(self.scheduler, EulerAncestralDiscreteScheduler):
                        z_ref_noised = (
                            alpha_hat**0.5 * z_ref + (1 - alpha_hat) ** 0.5 * noise
                        )
                    else:
                        z_ref_noised = self.scheduler.add_noise(z_ref, noise, t)

                    ref_prompt_embd = torch.cat([null_prompt_emb] * (len(posotive_latents) + len(negative_latents)), dim=0)

                    cached_hidden_states = self.get_unet_hidden_states(
                        z_ref_noised, t, ref_prompt_embd
                    )

                    n_pos, n_neg = positive_latents.shape[0], negative_latents.shape[0]
                    cached_pos_hs, cached_neg_hs = [], []
                    for hs in cached_hidden_states:
                        cached_pos, cached_neg = hs.split([n_pos, n_neg], dim=0)
                        cached_pos = cached_pos.view(
                            1, -1, *cached_pos.shape[2:]
                        ).expand(n_images, -1, -1)
                        cached_neg = cached_neg.view(
                            1, -1, *cached_neg.shape[2:]
                        ).expand(n_images, -1, -1)
                        cached_pos_hs.append(cached_pos)
                        cached_neg_hs.append(cached_neg)

                    if n_pos == 0:
                        cached_pos_hs = None
                    if n_neg == 0:
                        cached_neg_hs = None
                else:
                    cached_pos_hs, cached_neg_hs = None, None

                unet_out = self.unet_forward_with_cached_hidden_states(
                    z_all,
                    t,
                    prompt_embd=batched_prompt_embd,
                    cached_pos_hiddens=cached_pos_hs,
                    cached_neg_hiddens=cached_neg_hs,
                    pos_weights=pos_ws,
                    neg_weights=neg_ws,
                ).sample

                noise_cond, noise_uncond = unet_out.chunk(2)
                guidance = noise_cond - noise_uncond
                noise_pred = noise_uncond + guidance_scale * guidance
                latent_noise = self.scheduler.step(noise_pred, t, latent_noise).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    pbar.update()

        y = self.pipeline.decode_latents(latent_noise)
        imgs = self.pipeline.numpy_to_pil(y)

        return FabricPipelineOutput(imgs,False)

    @staticmethod
    def image_to_tensor(image: Union[str, Image.Image]):
        """
        Convert a PIL image to a torch tensor.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512, 512))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)


