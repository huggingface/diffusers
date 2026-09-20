# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch
import torch.nn.functional as F

from ...models import QwenImage21Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class QwenImage21TextInputsStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Expand prompt embeddings and vision masks to the requested output batch."

    @property
    def inputs(self):
        return [
            InputParam.template("prompt_embeds", required=True),
            InputParam.template("negative_prompt_embeds", required=False),
            InputParam.template("prompt_embeds_mask", required=False),
            InputParam.template("negative_prompt_embeds_mask", required=False),
            InputParam(
                "image_pad_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Positive prompt vision positions.",
            ),
            InputParam(
                "negative_image_pad_mask", type_hint=torch.Tensor, description="Negative prompt vision positions."
            ),
            InputParam.template("num_images_per_prompt", default=1),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds_mask"),
            OutputParam("image_pad_mask", type_hint=torch.Tensor, description="Expanded positive vision mask."),
            OutputParam(
                "negative_image_pad_mask", type_hint=torch.Tensor, description="Expanded negative vision mask."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        if not isinstance(block_state.num_images_per_prompt, int) or block_state.num_images_per_prompt < 1:
            raise ValueError("`num_images_per_prompt` must be a positive integer.")
        for name in (
            "prompt_embeds",
            "negative_prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds_mask",
            "image_pad_mask",
            "negative_image_pad_mask",
        ):
            value = getattr(block_state, name)
            if value is not None:
                value = value.repeat_interleave(block_state.num_images_per_prompt, dim=0).to(
                    components._execution_device
                )
                if name.endswith("embeds_mask") and value.all():
                    value = None
            setattr(block_state, name, value)
        self.set_block_state(state, block_state)
        return components, state


class QwenImage21PrepareLatentsStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Prepare target noise and spatial metadata for unpatched QwenImage21 latents."

    @property
    def expected_components(self):
        return [ComponentSpec("transformer", QwenImage21Transformer2DModel)]

    @property
    def inputs(self):
        return [
            InputParam.template("prompt_embeds", required=True),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam("output_resolution", default=1024, type_hint=int, description="Default output side length."),
            InputParam.template("generator"),
            InputParam.template("latents"),
            InputParam("condition_latents", type_hint=torch.Tensor, description="Packed condition tokens."),
            InputParam("condition_shapes", type_hint=list, description="Spatial shape of each condition image."),
            InputParam(
                "image_pad_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Positive prompt vision positions.",
            ),
            InputParam(
                "negative_image_pad_mask", type_hint=torch.Tensor, description="Negative prompt vision positions."
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Initial target noise."),
            OutputParam("height", type_hint=int, description="Output height in pixels."),
            OutputParam("width", type_hint=int, description="Output width in pixels."),
            OutputParam(
                "condition_latents",
                type_hint=torch.Tensor,
                description="Condition tokens expanded to the output batch.",
            ),
            OutputParam(
                "img_shapes", type_hint=list, description="Condition and target grid shapes for rotary embeddings."
            ),
            OutputParam(
                "img_mask", type_hint=torch.Tensor, description="Joint positive prompt and target vision positions."
            ),
            OutputParam(
                "negative_img_mask",
                type_hint=torch.Tensor,
                description="Joint negative prompt and target vision positions.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        height = block_state.height or block_state.output_resolution
        width = block_state.width or block_state.output_resolution
        if min(height, width) < 32 or height % 32 or width % 32:
            raise ValueError("`height` and `width` must be positive multiples of 32.")
        block_state.height, block_state.width = height, width
        batch = block_state.prompt_embeds.shape[0]
        channels = components.transformer.config.in_channels
        dtype, device = block_state.prompt_embeds.dtype, components._execution_device
        shape = (batch, 1, channels, height // 16, width // 16)
        if block_state.latents is None:
            noise = randn_tensor(shape, generator=block_state.generator, dtype=dtype, device=device)
            block_state.latents = noise.view(batch, channels, -1).transpose(1, 2)
        else:
            if block_state.latents.shape != (batch, height * width // 256, channels):
                raise ValueError("`latents` must have shape (effective batch, height * width / 256, latent channels).")
            block_state.latents = block_state.latents.to(device=device, dtype=dtype)
        if block_state.condition_latents is not None:
            block_state.condition_latents = block_state.condition_latents.to(device=device, dtype=dtype).expand(
                batch, -1, -1
            )
        block_state.img_shapes = [(block_state.condition_shapes or []) + [(1, height // 16, width // 16)]] * batch
        target_slots = block_state.latents.shape[1] // 4
        block_state.img_mask = torch.cat(
            [block_state.image_pad_mask, block_state.image_pad_mask.new_ones(batch, target_slots)], dim=1
        )
        block_state.negative_img_mask = None
        if block_state.negative_image_pad_mask is not None:
            block_state.negative_img_mask = torch.cat(
                [
                    block_state.negative_image_pad_mask,
                    block_state.negative_image_pad_mask.new_ones(batch, target_slots),
                ],
                dim=1,
            )
        self.set_block_state(state, block_state)
        return components, state


class QwenImage21SetTimestepsStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Set the flow-matching schedule using the target image sequence length."

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self):
        return [
            InputParam.template("num_inference_steps", default=40),
            InputParam.template("sigmas"),
            InputParam.template("latents", required=True),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="Denoising timesteps."),
            OutputParam("num_inference_steps", type_hint=int, description="Number of denoising timesteps."),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        if block_state.num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be positive.")
        config = components.scheduler.config
        base_len, max_len = config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 4096)
        base_shift, max_shift = config.get("base_shift", 0.5), config.get("max_shift", 1.15)
        m = (max_shift - base_shift) / (max_len - base_len)
        mu = block_state.latents.shape[1] * m + base_shift - m * base_len
        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps)
        components.scheduler.set_timesteps(sigmas=sigmas, device=components._execution_device, mu=mu)
        components.scheduler.set_begin_index(0)
        block_state.timesteps = components.scheduler.timesteps
        block_state.num_inference_steps = len(block_state.timesteps)
        self.set_block_state(state, block_state)
        return components, state


class QwenImage21PrepareInpaintStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Apply strength, retain source noise, and prepare the latent repaint mask."

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self):
        return [
            InputParam.template("latents", required=True),
            InputParam("source_latents", required=True, type_hint=torch.Tensor, description="Encoded source tokens."),
            InputParam("processed_mask", required=True, type_hint=torch.Tensor, description="Binary repaint mask."),
            InputParam("timesteps", required=True, type_hint=torch.Tensor, description="Full denoising schedule."),
            InputParam.template("num_inference_steps", default=40),
            InputParam.template("strength", default=1.0),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Source latents with initial noise."),
            OutputParam(
                "initial_noise", type_hint=torch.Tensor, description="Noise reused when preserving the source."
            ),
            OutputParam(
                "source_latents", type_hint=torch.Tensor, description="Source tokens expanded to the output batch."
            ),
            OutputParam("mask", type_hint=torch.Tensor, description="Repaint weights for target latent tokens."),
            OutputParam("timesteps", type_hint=torch.Tensor, description="Strength-adjusted denoising schedule."),
            OutputParam("num_inference_steps", type_hint=int, description="Number of retained steps."),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        if not 0 < block_state.strength <= 1:
            raise ValueError("`strength` must be in (0, 1].")
        count = int(block_state.num_inference_steps * block_state.strength)
        if count < 1:
            raise ValueError("`strength` leaves no denoising steps; increase strength or the step count.")
        start = block_state.num_inference_steps - count
        block_state.timesteps = block_state.timesteps[start:]
        block_state.num_inference_steps = count
        components.scheduler.set_begin_index(start)
        block_state.initial_noise = block_state.latents
        block_state.source_latents = block_state.source_latents.to(block_state.latents).expand(
            block_state.latents.shape[0], -1, -1
        )
        block_state.latents = components.scheduler.scale_noise(
            block_state.source_latents, block_state.timesteps[:1], block_state.initial_noise
        )
        mask = F.interpolate(
            block_state.processed_mask, size=(block_state.height // 16, block_state.width // 16), mode="nearest"
        )
        block_state.mask = mask.flatten(2).transpose(1, 2).to(block_state.latents)
        self.set_block_state(state, block_state)
        return components, state
