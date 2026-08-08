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

import math

import numpy as np
import torch
import torch.nn.functional as F

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import WanAnimate2Transformer3DModel
from ...schedulers import DPMSolverMultistepScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    SequentialPipelineBlocks,
)
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .decoders import WanVaeDecoderStep
from .encoders import WanTextEncoderStep
from .modular_pipeline import WanModularPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


class WanAnimate2ImageEncoderStep(ModularPipelineBlocks):
    """Encode reference image with CLIP + VAE, and driving video with VAE."""

    model_name = "wan"

    @property
    def expected_components(self):
        return [
            ComponentSpec("image_encoder", None),
            ComponentSpec("vae", None),
            ComponentSpec("image_processor", None),
        ]

    @property
    def description(self):
        return "Encode reference image (CLIP + VAE) and driving video (VAE) for Wan-Animate-2."

    @property
    def inputs(self):
        return [
            InputParam("image", required=True, type_hint=object, description="Reference character image."),
            InputParam("driving_video", required=True, type_hint=list, description="Driving video frames."),
            InputParam("height", required=True, type_hint=int),
            InputParam("width", required=True, type_hint=int),
            InputParam("prompt_ref", required=False, type_hint=str, default="人物动作的参考视频"),
        ]

    @property
    def outputs(self):
        return [
            OutputParam("clip_fea", type_hint=torch.Tensor, description="CLIP features of reference image."),
            OutputParam("ref_latents", type_hint=torch.Tensor, description="VAE latents of reference image."),
            OutputParam("condition_latents", type_hint=torch.Tensor, description="VAE latents of driving video."),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        device = state.device
        dtype = components.transformer.dtype

        # CLIP encode reference image
        image = components.image_processor(images=state.image, return_tensors="pt").to(device)
        image_embeds = components.image_encoder(**image, output_hidden_states=True)
        state.clip_fea = image_embeds.hidden_states[-2].to(dtype)

        # VAE encode reference image
        ref_pixels = components.image_processor(images=state.image, return_tensors="pt").to(
            device=device, dtype=components.vae.dtype
        )
        ref_latents = components.vae.encode(ref_pixels)
        latents_mean = torch.tensor(components.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(ref_latents)
        latents_recip_std = 1.0 / torch.tensor(components.vae.config.latents_std).view(1, -1, 1, 1, 1).to(ref_latents)
        state.ref_latents = (ref_latents - latents_mean) * latents_recip_std

        # VAE encode driving video
        driving_pixels = state.driving_video.to(device=device, dtype=components.vae.dtype)
        condition_latents = components.vae.encode(driving_pixels)
        state.condition_latents = (condition_latents - latents_mean) * latents_recip_std

        return components, state


class WanAnimate2SetTimestepsStep(ModularPipelineBlocks):
    """Set timesteps using custom sigma computation for flow matching."""

    model_name = "wan"

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", DPMSolverMultistepScheduler)]

    @property
    def description(self):
        return "Set timesteps for Wan-Animate-2 with custom sigmas."

    @property
    def inputs(self):
        return [
            InputParam("num_inference_steps", required=True, type_hint=int),
            InputParam("sample_shift", required=False, type_hint=float, default=5.0),
        ]

    @property
    def outputs(self):
        return [OutputParam("timesteps", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state):
        sigmas = _get_sampling_sigmas(state.num_inference_steps, state.sample_shift)
        components.scheduler.set_timesteps(sigmas=sigmas, device=state.device)
        state.timesteps = components.scheduler.timesteps
        return components, state


class WanAnimate2PrepareLatentsStep(ModularPipelineBlocks):
    """Prepare noise latents and encode reference (forward_ref -> KV cache)."""

    model_name = "wan"

    @property
    def expected_components(self):
        return [ComponentSpec("transformer", WanAnimate2Transformer3DModel)]

    @property
    def description(self):
        return "Prepare noise latents and encode reference video to cache KV."

    @property
    def inputs(self):
        return [
            InputParam("ref_latents", required=True, type_hint=torch.Tensor),
            InputParam("clip_fea_ref", required=True, type_hint=torch.Tensor),
            InputParam("condition_latents", required=True, type_hint=torch.Tensor),
            InputParam("prompt_ref_embeds", required=True, type_hint=torch.Tensor),
            InputParam("height", required=True, type_hint=int),
            InputParam("width", required=True, type_hint=int),
            InputParam("clip_len", required=True, type_hint=int),
            InputParam("generator", required=False, type_hint=torch.Generator),
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
        ]

    @property
    def outputs(self):
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("k_cache", type_hint=dict),
            OutputParam("v_cache", type_hint=dict),
            OutputParam("grid_sizes_ref", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        device = state.device
        dtype = components.transformer.dtype

        latent_h = state.height // 8
        latent_w = state.width // 8
        clip_len = state.clip_len
        lat_t = (clip_len + 3) // 4 + 1

        # Prepare noise
        noise = torch.randn(
            16, lat_t, latent_h, latent_w, device=device, dtype=torch.float32, generator=state.generator
        )
        state.latents = [noise]

        # Prepare grid sizes for reference
        ref_shape = state.condition_latents.shape[2:]
        state.grid_sizes_ref = torch.tensor([ref_shape], dtype=torch.long)

        # KV cache
        state.k_cache = {}
        state.v_cache = {}

        # Reference encoding (forward_ref)
        max_seq_len_ref = int(math.ceil(np.prod(ref_shape)))

        # Prepare y_ref (mask + ref_latents)
        mask_ref = torch.zeros(1, 4, *state.ref_latents.shape[2:], device=device, dtype=dtype)
        mask_ref[:, :, 0:1] = 1
        mask_ref = mask_ref.view(1, -1, 4, *state.ref_latents.shape[2:]).transpose(1, 2).squeeze(0)
        y_ref = torch.cat([mask_ref, state.ref_latents[0]], dim=0)

        t_ref = torch.tensor([state.timesteps[0].item()], device=device, dtype=dtype)

        components.transformer(
            [state.ref_latents[0]],
            grid_sizes=state.grid_sizes_ref,
            k_cache=state.k_cache,
            v_cache=state.v_cache,
            clip_fea_ref=state.clip_fea_ref,
            y_ref=[y_ref],
            context_ref=[state.prompt_ref_embeds[0]],
            seq_len_ref=max_seq_len_ref,
            t=t_ref,
            method="forward_ref",
        )

        return components, state


class WanAnimate2LoopBeforeDenoiser(ModularPipelineBlocks):
    """Prepare latent model input for the denoiser."""

    model_name = "wan"

    @property
    def description(self):
        return "Prepare latent model input within the denoising loop."

    @property
    def inputs(self):
        return [InputParam("latents", required=True, type_hint=list)]

    @torch.no_grad()
    def __call__(self, components, state, i, t):
        state.latent_model_input = state.latents[0]
        return components, state


class WanAnimate2LoopDenoiser(ModularPipelineBlocks):
    """Denoiser that calls forward_gen with cached KV."""

    model_name = "wan"

    def __init__(self, guider_input_fields=None):
        if guider_input_fields is None:
            guider_input_fields = {"context": ("prompt_embeds", "negative_prompt_embeds")}
        self._guider_input_fields = guider_input_fields
        super().__init__()

    @property
    def expected_components(self):
        return [
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 3.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", WanAnimate2Transformer3DModel),
        ]

    @property
    def description(self):
        return "Denoiser step that calls forward_gen with cached KV for Wan-Animate-2."

    @property
    def inputs(self):
        inputs = [InputParam("num_inference_steps", required=True, type_hint=int)]
        guider_names = []
        for v in self._guider_input_fields.values():
            if isinstance(v, tuple):
                guider_names.extend(v)
            else:
                guider_names.append(v)
        for name in guider_names:
            inputs.append(InputParam(name=name, required=True, type_hint=torch.Tensor))
        return inputs

    @torch.no_grad()
    def __call__(self, components, state, i, t):
        components.guider.set_state(step=i, num_inference_steps=state.num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs_from_block_state(state, self._guider_input_fields)

        for batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = batch.as_dict()
            cond_kwargs = {
                k: v.to(state.dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond_kwargs.items()
                if k in self._guider_input_fields
            }

            is_uncond = batch.guidance_identifier == "pred_uncond"
            batch.noise_pred = components.transformer(
                state.latents,
                k_cache=state.k_cache,
                v_cache=state.v_cache,
                clip_fea=state.clip_fea,
                y=state.y,
                seq_len=state.max_seq_len,
                t=t.expand(1),
                grid_sizes_ref=state.grid_sizes_ref,
                origin_len=state.origin_len,
                origin_area=state.origin_area,
                method="forward_gen",
                is_uncondtion=is_uncond,
                **cond_kwargs,
            )
            if isinstance(batch.noise_pred, list):
                batch.noise_pred = batch.noise_pred[0]
            components.guider.cleanup_models(components.transformer)

        state.noise_pred = components.guider(guider_state)[0]
        return components, state


class WanAnimate2DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    """Denoise loop for Wan-Animate-2: before_denoiser -> denoiser -> after_denoiser (scheduler step)."""

    model_name = "wan"
    sub_blocks = [WanAnimate2LoopBeforeDenoiser, WanAnimate2LoopDenoiser]

    @property
    def description(self):
        return "Denoise loop for Wan-Animate-2 using forward_gen with cached KV."

    @property
    def inputs(self):
        return [
            InputParam("latents", required=True, type_hint=list),
            InputParam("k_cache", required=True, type_hint=dict),
            InputParam("v_cache", required=True, type_hint=dict),
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
        ]

    @property
    def outputs(self):
        return [OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def after_denoiser(self, components, state, i, t):
        temp_x0 = components.scheduler.step(
            state.noise_pred.unsqueeze(0),
            t,
            state.latents[0].unsqueeze(0),
            return_dict=False,
        )[0]
        state.latents[0] = temp_x0.squeeze(0)
        return components, state


# ====================
# 1. CORE DENOISE
# ====================


# auto_docstring
class WanAnimate2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise block for Wan-Animate-2: set_timesteps -> prepare_latents (with ref encoding) -> denoise loop.

      Components:
          transformer (`WanAnimate2Transformer3DModel`) scheduler (`DPMSolverMultistepScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_inference_steps (`int`): Number of denoising steps.
          sample_shift (`float`): Shift for sigma computation.
          ref_latents (`Tensor`): VAE latents of reference image.
          condition_latents (`Tensor`): VAE latents of driving video.
          clip_fea (`Tensor`): CLIP features of reference image.
          clip_fea_ref (`Tensor`): CLIP features of driving video.
          prompt_embeds (`Tensor`): Text embeddings.
          negative_prompt_embeds (`Tensor`): Negative text embeddings.
          prompt_ref_embeds (`Tensor`): Reference text embeddings.
          height (`int`): Output height.
          width (`int`): Output width.
          clip_len (`int`): Frames per segment.
          generator (`Generator`): Random generator.

      Outputs:
          latents (`Tensor`): Denoised latents.
    """

    model_name = "wan"
    block_classes = [
        WanAnimate2SetTimestepsStep,
        WanAnimate2PrepareLatentsStep,
        WanAnimate2DenoiseLoopWrapper,
    ]
    block_names = ["set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Core denoise block for Wan-Animate-2."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# ====================
# 2. FULL BLOCKS
# ====================


# auto_docstring
class WanAnimate2Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline for character animation using Wan-Animate-2.

      Components:
          text_encoder (`UMT5EncoderModel`) tokenizer (`AutoTokenizer`) image_encoder (`CLIPVisionModel`)
          transformer (`WanAnimate2Transformer3DModel`) scheduler (`DPMSolverMultistepScheduler`) guider
          (`ClassifierFreeGuidance`) vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          prompt (`str`): Text prompt describing the character.
          negative_prompt (`str`): Negative prompt.
          prompt_ref (`str`): Reference prompt for driving video.
          image (`PIL.Image`): Reference character image.
          driving_video (`list`): Driving video frames.
          height (`int`): Output height.
          width (`int`): Output width.
          clip_len (`int`): Frames per segment.
          num_inference_steps (`int`): Number of denoising steps.
          sample_shift (`float`): Shift for sigma computation.
          generator (`Generator`): Random generator.
          output_type (`str`): Output format.

      Outputs:
          videos (`list`): The generated videos.
    """

    model_name = "wan"
    block_classes = [
        WanTextEncoderStep,
        WanAnimate2ImageEncoderStep,
        WanAnimate2CoreDenoiseStep,
        WanVaeDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "image_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return "Modular pipeline for character animation using Wan-Animate-2."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
