# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import List, Optional, Tuple, Union, Callable
from einops import rearrange

import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint


from ...models import VQModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import PaellaScheduler


class PaellaTextToImagePipeline(DiffusionPipeline):
    # TODO fix dymmmy docstring
    r"""
    Pipeline for text-to-image generation using Paella

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        ...
    """

    # vqvae: VQModel
    # text_encoder: CLIPTextModel
    # tokenizer: CLIPTokenizer
    # unet: DenoiseUNet
    # scheduler: PaellaScheduler

    # def __init__(
    #     self,
    #     vqvae: VQModel,
    #     text_encoder: CLIPTextModel,
    #     tokenizer: CLIPTokenizer,
    #     unet: DenoiseUNet,
    #     scheduler: PaellaScheduler,
    # ):
    def __init__(
        self,
        vqvae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

    def _encode_prompt(self, prompt, num_images_per_prompt):
        if isinstance(prompt, list):
            prompt = [one_prompt for one_prompt in prompt for _ in range(num_images_per_prompt)]
        else:
            prompt = [prompt] * num_images_per_prompt

        # TODO self.text_encoder should be clip_model.encode_text
        # TODO self.tokenizer should be tokenizer.tokenize
        # TODO why not text_input_ids?
        # get prompt text embeddings
        tokenized_text = self.tokenizer(prompt).to(self.device)
        text_embeddings = self.text_encoder(tokenized_text)

        # note: repeat_interleave is wrong
        # duplicate text embeddings for each generation per prompt
        # text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
        return text_embeddings

    def decode_latents(self, img_seq, latents_shape=(32, 32)):
        img_seq = img_seq.view(img_seq.shape[0], -1)
        b, n = img_seq.shape
        one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=self.vqvae.num_tokens).float()
        z = one_hot_indices @ self.vqvae.model.quantize.embed.weight
        z = rearrange(z, "b (h w) c -> b c h w", h=latents_shape[0], w=latents_shape[1])
        img = self.vqvae.model.decode(z)
        img = (img.clamp(-1.0, 1.0) + 1) * 0.5
        return img

    def prepare_latents(self, batch_size, latents_shape, device, latents=None, mask=None):
        size = (batch_size, *latents_shape)

        # If no starting image is provided, generate a random image
        if latents is None:
            latents = torch.randint(
                0,
                self.unet.num_vec_classes,
                size=size,
                device=device,
            )
        # If a mask is provided, apply it to the image
        elif mask is not None:
            latents = latents.to(device)
            mask = mask.to(device)
            noise = torch.randint(
                0,
                self.unet.num_vec_classes,
                size=size,
                device=device,
            )
            latents = noise * mask + (1 - mask) * latents

        start_latents = latents.clone()
        return latents, start_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        latents: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        latents_shape: Optional[Tuple[int, int]] = (32, 32),
        num_inference_steps: int = 12,
        guidance_scale: float = 5,
        num_images_per_prompt: Optional[int] = 1,
        renoise_mode: str = "start",
        temperature_range: Tuple[float, float] = [1.0, 1.0],
        do_locally_typical_sampling: bool = True,
        typical_mass: float = 0.2,
        typical_min_tokens: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # TODO add docstring
        """
        mask: a tensor of binary masks indicating which elements of x should be preserved (optional, default is None)
        temperature_range: a list of two values indicating the range of temperatures to use for the Gumbel-Softmax sampling (optional, default is [1.0, 1.0])
        do_locally_typical_sampling: whether to use locally typical sampling (optional, default is True)
        typical_mass: a value in the range [0, 1] indicating the mass of the typical set to preserve
        """
        # 0. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = batch_size * num_images_per_prompt

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, num_images_per_prompt)
        assert text_embeddings.size(0) == batch_size

        # 2. Prepare timesteps, temperature
        self.scheduler.set_temperatures(num_inference_steps, batch_size, temperature_range, device=self.device)
        temperatures = self.scheduler.temperatures
        rs = self.scheduler.rs

        # 3. Prepare latent variables
        latents, start_latents = self.prepare_latents(batch_size, latents_shape, self.device, latents, mask)
        random_noise_to_renoise = start_latents if renoise_mode == "start" else None

        # 4. Denoising loop
        for idx, r in enumerate(self.progress_bar(rs)):
            if renoise_mode == "previous":
                random_noise_to_renoise = latents.clone()

            # predict the previous noisy latents x_t -> x_t-1
            # latents have a shape of h × w × N where N is the number of codebook items
            latents_text = self.unet(latents, text_embeddings, r)

            if do_classifier_free_guidance:
                latents_uncond = self.unet(latents, torch.zeros_like(text_embeddings), r)
                latents_text = torch.lerp(latents_uncond, latents_text, guidance_scale)
            latents = latents_text

            do_renoise = idx < num_inference_steps - 1
            latents = self.scheduler.step(
                idx,
                latents,
                mask,
                temperatures[idx],
                do_locally_typical_sampling,
                typical_mass,
                typical_min_tokens,
                do_renoise,
                random_noise_to_renoise,
                start_latents,
            ).prev_sample

            # call the callback, if provided
            if callback is not None and idx % callback_steps == 0:
                callback(idx, r, temperatures[idx], latents)

        # 5. Post-processing
        latents = latents.detach()
        # decode the image latents with the VQVAE
        image = self.decode_latents(latents, latents_shape)
        
        # NOTE fix output type
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


# Copied from Paella/modules.py
class ModulatedLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, channels_first=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, 1, 1))
        self.channels_first = channels_first

    def forward(self, x, w=None):
        x = x.permute(0, 2, 3, 1) if self.channels_first else x
        if w is None:
            x = self.ln(x)
        else:
            x = self.gamma * w * self.ln(x) + self.beta * w
        x = x.permute(0, 3, 1, 2) if self.channels_first else x
        return x


class ResBlock(nn.Module):
    def __init__(self, c, c_hidden, c_cond=0, c_skip=0, scaler=None, layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, kernel_size=3, groups=c))
        self.ln = ModulatedLayerNorm(c, channels_first=False)
        self.channelwise = nn.Sequential(
            nn.Linear(c + c_skip, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(c), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.scaler = scaler
        if c_cond > 0:
            self.cond_mapper = nn.Linear(c_cond, c)

    def forward(self, x, s=None, skip=None):
        res = x
        x = self.depthwise(x)
        if s is not None:
            if s.size(2) == s.size(3) == 1:
                s = s.expand(-1, -1, x.size(2), x.size(3))
            elif s.size(2) != x.size(2) or s.size(3) != x.size(3):
                s = nn.functional.interpolate(s, size=x.shape[-2:], mode="bilinear")
            s = self.cond_mapper(s.permute(0, 2, 3, 1))
            # s = self.cond_mapper(s.permute(0, 2, 3, 1))
            # if s.size(1) == s.size(2) == 1:
            #     s = s.expand(-1, x.size(2), x.size(3), -1)
        x = self.ln(x.permute(0, 2, 3, 1), s)
        if skip is not None:
            x = torch.cat([x, skip.permute(0, 2, 3, 1)], dim=-1)
        x = self.channelwise(x)
        x = self.gamma * x if self.gamma is not None else x
        x = res + x.permute(0, 3, 1, 2)
        if self.scaler is not None:
            x = self.scaler(x)
        return x


class DenoiseUNet(nn.Module):
    def __init__(
        self,
        num_vec_classes,
        c_hidden=1280,
        c_clip=1024,
        c_r=64,
        down_levels=[4, 8, 16],
        up_levels=[16, 8, 4],
    ):
        super().__init__()
        self.num_vec_classes = num_vec_classes
        self.c_r = c_r
        self.down_levels = down_levels
        self.up_levels = up_levels
        c_levels = [c_hidden // (2**i) for i in reversed(range(len(down_levels)))]
        self.embedding = nn.Embedding(num_vec_classes, c_levels[0])

        # DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(down_levels):
            blocks = []
            if i > 0:
                blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            for _ in range(num_blocks):
                block = ResBlock(c_levels[i], c_levels[i] * 4, c_clip + c_r)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)
            self.down_blocks.append(nn.ModuleList(blocks))

        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            for j in range(num_blocks):
                block = ResBlock(
                    c_levels[len(c_levels) - 1 - i],
                    c_levels[len(c_levels) - 1 - i] * 4,
                    c_clip + c_r,
                    c_levels[len(c_levels) - 1 - i] if (j == 0 and i > 0) else 0,
                )
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(up_levels))
                blocks.append(block)
            if i < len(up_levels) - 1:
                blocks.append(
                    nn.ConvTranspose2d(
                        c_levels[len(c_levels) - 1 - i],
                        c_levels[len(c_levels) - 2 - i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
            self.up_blocks.append(nn.ModuleList(blocks))

        self.clf = nn.Conv2d(c_levels[0], num_vec_classes, kernel_size=1)

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def gen_r_embedding(self, r, max_positions=10000):
        dtype = r.dtype
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb.to(dtype)

    def _down_encode_(self, x, s):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    # s_level = s[:, 0]
                    # s = s[:, 1:]
                    x = block(x, s)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, s):
        x = level_outputs[0]
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    # s_level = s[:, 0]
                    # s = s[:, 1:]
                    if i > 0 and j == 0:
                        x = block(x, s, level_outputs[i])
                    else:
                        x = block(x, s)
                else:
                    x = block(x)
        return x

    def forward(self, x, c, r):  # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x).permute(0, 3, 1, 2)
        if len(c.shape) == 2:
            s = torch.cat([c, r_embed], dim=-1)[:, :, None, None]
        else:
            r_embed = r_embed[:, :, None, None].expand(-1, -1, c.size(2), c.size(3))
            s = torch.cat([c, r_embed], dim=1)
        level_outputs = self._down_encode_(x, s)
        x = self._up_decode(level_outputs, s)
        x = self.clf(x)
        return x
