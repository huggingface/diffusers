# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import math

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance, ClassifierFreeZeroStarGuidance
from ...models import HeliosTransformer3DModel
from ...schedulers import HeliosScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .before_denoise import calculate_shift
from .modular_pipeline import HeliosModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def sample_block_noise(
    batch_size,
    channel,
    num_frames,
    height,
    width,
    gamma,
    patch_size=(1, 2, 2),
    device=None,
    generator=None,
):
    """Generate spatially-correlated block noise for pyramid upsampling correction.

    Uses a multivariate normal distribution with covariance based on `gamma` to produce noise with block structure,
    matching the upsampling artifacts that need correction.
    """
    # NOTE: A generator must be provided to ensure correct and reproducible results.
    # Creating a default generator here is a fallback only — without a fixed seed,
    # the output will be non-deterministic and may produce incorrect results in CP context.
    if generator is None:
        generator = torch.Generator(device=device)
    elif isinstance(generator, list):
        generator = generator[0]

    _, ph, pw = patch_size
    block_size = ph * pw

    cov = (
        torch.eye(block_size, device=device) * (1 + gamma) - torch.ones(block_size, block_size, device=device) * gamma
    )
    cov += torch.eye(block_size, device=device) * 1e-8
    cov = cov.float()  # Upcast to fp32 for numerical stability — cholesky is unreliable in fp16/bf16.

    L = torch.linalg.cholesky(cov)
    block_number = batch_size * channel * num_frames * (height // ph) * (width // pw)
    z = torch.randn(block_number, block_size, device=generator.device, generator=generator).to(device)
    noise = z @ L.T

    noise = noise.view(batch_size, channel, num_frames, height // ph, width // pw, ph, pw)
    noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)
    return noise


# ========================================
# Chunk Loop Leaf Blocks
# ========================================


class HeliosChunkHistorySliceStep(ModularPipelineBlocks):
    """Slices history latents into short/mid/long for a T2V chunk.

    At k==0 with no image_latents, creates a zero prefix. Otherwise uses image_latents (either provided or captured
    from first chunk by HeliosChunkUpdateStep).
    """

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "T2V history slice: splits history into long/mid/short. At k==0 with no image_latents, "
            "creates a zero prefix; otherwise uses image_latents as prefix for short history."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "keep_first_frame",
                default=True,
                type_hint=bool,
                description="Whether to keep the first frame as a prefix in history.",
            ),
            InputParam(
                "history_sizes",
                required=True,
                type_hint=list,
                description="Sizes of long/mid/short history buffers for temporal context.",
            ),
            InputParam(
                "history_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Accumulated history latents from previous chunks.",
            ),
            InputParam("latent_shape", required=True, type_hint=tuple),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        keep_first_frame = block_state.keep_first_frame
        history_sizes = block_state.history_sizes
        image_latents = block_state.image_latents
        device = components._execution_device

        batch_size, num_channels_latents, _, h_latent, w_latent = block_state.latent_shape

        if keep_first_frame:
            latents_history_long, latents_history_mid, latents_history_1x = block_state.history_latents[
                :, :, -sum(history_sizes) :
            ].split(history_sizes, dim=2)
            if image_latents is None and k == 0:
                latents_prefix = torch.zeros(
                    batch_size,
                    num_channels_latents,
                    1,
                    h_latent,
                    w_latent,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                latents_prefix = image_latents
            latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
        else:
            latents_history_long, latents_history_mid, latents_history_short = block_state.history_latents[
                :, :, -sum(history_sizes) :
            ].split(history_sizes, dim=2)

        block_state.latents_history_short = latents_history_short
        block_state.latents_history_mid = latents_history_mid
        block_state.latents_history_long = latents_history_long

        return components, block_state


class HeliosI2VChunkHistorySliceStep(ModularPipelineBlocks):
    """Slices history latents into short/mid/long for an I2V chunk.

    Always uses image_latents as prefix (assumes history pre-seeded with fake_image_latents).
    """

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "I2V history slice: splits pre-seeded history into long/mid/short, "
            "always using image_latents as prefix for short history."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "keep_first_frame",
                default=True,
                type_hint=bool,
                description="Whether to keep the first frame as a prefix in history.",
            ),
            InputParam(
                "history_sizes",
                required=True,
                type_hint=list,
                description="Sizes of long/mid/short history buffers for temporal context.",
            ),
            InputParam(
                "history_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Accumulated history latents from previous chunks.",
            ),
            InputParam(
                "image_latents",
                required=True,
                type_hint=torch.Tensor,
                description="First-frame latents used as prefix for short history.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        keep_first_frame = block_state.keep_first_frame
        history_sizes = block_state.history_sizes
        image_latents = block_state.image_latents

        if keep_first_frame:
            latents_history_long, latents_history_mid, latents_history_1x = block_state.history_latents[
                :, :, -sum(history_sizes) :
            ].split(history_sizes, dim=2)
            latents_history_short = torch.cat([image_latents, latents_history_1x], dim=2)
        else:
            latents_history_long, latents_history_mid, latents_history_short = block_state.history_latents[
                :, :, -sum(history_sizes) :
            ].split(history_sizes, dim=2)

        block_state.latents_history_short = latents_history_short
        block_state.latents_history_mid = latents_history_mid
        block_state.latents_history_long = latents_history_long

        return components, block_state


class HeliosChunkNoiseGenStep(ModularPipelineBlocks):
    """Generates noise latents for a chunk using randn_tensor."""

    model_name = "helios"

    @property
    def description(self) -> str:
        return "Generates random noise latents at full resolution for a single chunk."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latent_shape", required=True, type_hint=tuple),
            InputParam.template("generator"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        device = components._execution_device
        block_state.latents = randn_tensor(
            block_state.latent_shape, generator=block_state.generator, device=device, dtype=torch.float32
        )
        return components, block_state


class HeliosPyramidChunkNoiseGenStep(ModularPipelineBlocks):
    """Generates noise latents and downsamples to smallest pyramid level."""

    model_name = "helios-pyramid"

    @property
    def description(self) -> str:
        return (
            "Generates random noise at full resolution, then downsamples to the smallest "
            "pyramid level via bilinear interpolation."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latent_shape", required=True, type_hint=tuple),
            InputParam(
                "pyramid_num_inference_steps_list",
                default=[10, 10, 10],
                type_hint=list,
                description="Number of denoising steps per pyramid stage.",
            ),
            InputParam.template("generator"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        device = components._execution_device
        batch_size, num_channels_latents, num_latent_frames, h_latent, w_latent = block_state.latent_shape

        latents = randn_tensor(
            block_state.latent_shape, generator=block_state.generator, device=device, dtype=torch.float32
        )

        # Downsample to smallest pyramid level
        h, w = h_latent, w_latent
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_latent_frames, num_channels_latents, h, w)
        for _ in range(len(block_state.pyramid_num_inference_steps_list) - 1):
            h //= 2
            w //= 2
            latents = F.interpolate(latents, size=(h, w), mode="bilinear") * 2
        block_state.latents = latents.reshape(batch_size, num_latent_frames, num_channels_latents, h, w).permute(
            0, 2, 1, 3, 4
        )

        return components, block_state


class HeliosChunkSchedulerResetStep(ModularPipelineBlocks):
    """Resets the scheduler with timesteps for a single chunk."""

    model_name = "helios"

    @property
    def description(self) -> str:
        return "Resets the scheduler with the correct timesteps and shift parameter (mu) for this chunk."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", HeliosScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("mu", required=True, type_hint=float),
            InputParam.template("sigmas", required=True),
            InputParam.template("num_inference_steps"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        device = components._execution_device
        components.scheduler.set_timesteps(
            block_state.num_inference_steps, device=device, sigmas=block_state.sigmas, mu=block_state.mu
        )
        block_state.timesteps = components.scheduler.timesteps

        return components, block_state


# ========================================
# Inner Denoising Blocks
# ========================================


class HeliosChunkDenoiseInner(ModularPipelineBlocks):
    """Inner timestep loop for denoising a single chunk, using guider for guidance."""

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Inner denoising loop that iterates over timesteps for a single chunk. "
            "Uses the guider to manage conditional/unconditional forward passes with cache_context, "
            "applies guidance, and runs scheduler step."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HeliosTransformer3DModel),
            ComponentSpec("scheduler", HeliosScheduler),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam.template("timesteps"),
            InputParam("prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam.template("denoiser_input_fields"),
            InputParam.template("num_inference_steps"),
            InputParam.template("attention_kwargs"),
            InputParam.template("generator"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        latents = block_state.latents
        timesteps = block_state.timesteps
        num_inference_steps = block_state.num_inference_steps

        transformer_dtype = components.transformer.dtype
        num_warmup_steps = len(timesteps) - num_inference_steps * components.scheduler.order

        # Guider inputs: only encoder_hidden_states differs between cond/uncond
        guider_inputs = {
            "encoder_hidden_states": (block_state.prompt_embeds, block_state.negative_prompt_embeds),
        }

        # Build shared kwargs from denoiser_input_fields (excludes guider-managed ones)
        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        shared_kwargs = {}
        for field_name, field_value in block_state.denoiser_input_fields.items():
            if field_name in transformer_args and field_name not in guider_inputs:
                shared_kwargs[field_name] = field_value

        # Add loop-internal history latents with dtype casting
        shared_kwargs["latents_history_short"] = block_state.latents_history_short.to(transformer_dtype)
        shared_kwargs["latents_history_mid"] = block_state.latents_history_mid.to(transformer_dtype)
        shared_kwargs["latents_history_long"] = block_state.latents_history_long.to(transformer_dtype)
        shared_kwargs["attention_kwargs"] = block_state.attention_kwargs

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(torch.int64)
                latent_model_input = latents.to(transformer_dtype)

                components.guider.set_state(step=i, num_inference_steps=num_inference_steps, timestep=t)
                guider_state = components.guider.prepare_inputs(guider_inputs)

                for guider_state_batch in guider_state:
                    components.guider.prepare_models(components.transformer)
                    cond_kwargs = {k: getattr(guider_state_batch, k) for k in guider_inputs.keys()}

                    context_name = getattr(guider_state_batch, components.guider._identifier_key)
                    with components.transformer.cache_context(context_name):
                        guider_state_batch.noise_pred = components.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            return_dict=False,
                            **cond_kwargs,
                            **shared_kwargs,
                        )[0]
                    components.guider.cleanup_models(components.transformer)

                noise_pred = components.guider(guider_state)[0]

                # Scheduler step
                latents = components.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    generator=block_state.generator,
                    return_dict=False,
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        block_state.latents = latents
        return components, block_state


class HeliosPyramidChunkDenoiseInner(ModularPipelineBlocks):
    """Nested pyramid stage loop with inner timestep denoising.

    For each pyramid stage (small -> full resolution):
    1. Upsample latents + block noise correction (stages > 0)
    2. Compute mu from current resolution, set scheduler timesteps
    3. Run timestep denoising loop (same logic as HeliosChunkDenoiseInner)
    """

    model_name = "helios-pyramid"

    @property
    def description(self) -> str:
        return (
            "Pyramid denoising inner block: loops over pyramid stages from smallest to full resolution. "
            "Each stage upsamples latents (with block noise correction), recomputes scheduler parameters, "
            "and runs the timestep denoising loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HeliosTransformer3DModel),
            ComponentSpec("scheduler", HeliosScheduler),
            ComponentSpec(
                "guider",
                ClassifierFreeZeroStarGuidance,
                config=FrozenDict({"guidance_scale": 5.0, "zero_init_steps": 2}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam("prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam.template("denoiser_input_fields"),
            InputParam(
                "pyramid_num_inference_steps_list",
                default=[10, 10, 10],
                type_hint=list,
                description="Number of denoising steps per pyramid stage.",
            ),
            InputParam.template("attention_kwargs"),
            InputParam.template("generator"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        device = components._execution_device
        transformer_dtype = components.transformer.dtype
        latents = block_state.latents
        pyramid_num_stages = len(block_state.pyramid_num_inference_steps_list)

        # Guider inputs: only encoder_hidden_states differs between cond/uncond
        guider_inputs = {
            "encoder_hidden_states": (block_state.prompt_embeds, block_state.negative_prompt_embeds),
        }

        # Build shared kwargs from denoiser_input_fields (excludes guider-managed ones)
        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        shared_kwargs = {}
        for field_name, field_value in block_state.denoiser_input_fields.items():
            if field_name in transformer_args and field_name not in guider_inputs:
                shared_kwargs[field_name] = field_value

        # Add loop-internal history latents with dtype casting
        shared_kwargs["latents_history_short"] = block_state.latents_history_short.to(transformer_dtype)
        shared_kwargs["latents_history_mid"] = block_state.latents_history_mid.to(transformer_dtype)
        shared_kwargs["latents_history_long"] = block_state.latents_history_long.to(transformer_dtype)
        shared_kwargs["attention_kwargs"] = block_state.attention_kwargs

        # Save original zero_init_steps if the guider supports it (e.g. ClassifierFreeZeroStarGuidance).
        # Helios only applies zero init in pyramid stage 0 (lowest resolution), so we disable it
        # for subsequent stages by temporarily setting zero_init_steps=0.
        orig_zero_init_steps = getattr(components.guider, "zero_init_steps", None)

        for i_s in range(pyramid_num_stages):
            # --- Stage setup ---

            # Disable zero init for stages > 0 (only stage 0 should have zero init)
            if orig_zero_init_steps is not None and i_s > 0:
                components.guider.zero_init_steps = 0

            # a. Compute mu from current resolution (before upsample, matching standard pipeline)
            patch_size = components.transformer.config.patch_size
            image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                patch_size[0] * patch_size[1] * patch_size[2]
            )
            mu = calculate_shift(
                image_seq_len,
                components.scheduler.config.get("base_image_seq_len", 256),
                components.scheduler.config.get("max_image_seq_len", 4096),
                components.scheduler.config.get("base_shift", 0.5),
                components.scheduler.config.get("max_shift", 1.15),
            )

            # b. Set scheduler timesteps for this stage
            num_inference_steps = block_state.pyramid_num_inference_steps_list[i_s]
            components.scheduler.set_timesteps(
                num_inference_steps,
                i_s,
                device=device,
                mu=mu,
            )
            timesteps = components.scheduler.timesteps

            # c. Upsample + block noise correction for stages > 0
            if i_s > 0:
                batch_size, num_channels_latents, num_frames, current_h, current_w = latents.shape
                new_h = current_h * 2
                new_w = current_w * 2

                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_frames, num_channels_latents, current_h, current_w
                )
                latents = F.interpolate(latents, size=(new_h, new_w), mode="nearest")
                latents = latents.reshape(batch_size, num_frames, num_channels_latents, new_h, new_w).permute(
                    0, 2, 1, 3, 4
                )

                # Block noise correction
                ori_sigma = 1 - components.scheduler.ori_start_sigmas[i_s]
                gamma = components.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                batch_size, num_channels_latents, num_frames, h, w = latents.shape
                noise = sample_block_noise(
                    batch_size,
                    num_channels_latents,
                    num_frames,
                    h,
                    w,
                    gamma,
                    patch_size,
                    device=device,
                    generator=block_state.generator,
                )
                noise = noise.to(dtype=transformer_dtype)
                latents = alpha * latents + beta * noise

            # --- Timestep denoising loop ---
            num_warmup_steps = len(timesteps) - num_inference_steps * components.scheduler.order

            with tqdm(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    timestep = t.expand(latents.shape[0]).to(torch.int64)
                    latent_model_input = latents.to(transformer_dtype)

                    components.guider.set_state(step=i, num_inference_steps=num_inference_steps, timestep=t)
                    guider_state = components.guider.prepare_inputs(guider_inputs)

                    for guider_state_batch in guider_state:
                        components.guider.prepare_models(components.transformer)
                        cond_kwargs = {kk: getattr(guider_state_batch, kk) for kk in guider_inputs.keys()}

                        context_name = getattr(guider_state_batch, components.guider._identifier_key)
                        with components.transformer.cache_context(context_name):
                            guider_state_batch.noise_pred = components.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                return_dict=False,
                                **cond_kwargs,
                                **shared_kwargs,
                            )[0]
                        components.guider.cleanup_models(components.transformer)

                    noise_pred = components.guider(guider_state)[0]

                    # Scheduler step
                    latents = components.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        generator=block_state.generator,
                        return_dict=False,
                    )[0]

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % components.scheduler.order == 0
                    ):
                        progress_bar.update()

        # Restore original zero_init_steps
        if orig_zero_init_steps is not None:
            components.guider.zero_init_steps = orig_zero_init_steps

        block_state.latents = latents
        return components, block_state


# ========================================
# Post-Denoise Update
# ========================================


class HeliosChunkUpdateStep(ModularPipelineBlocks):
    """Updates chunk collection and history after denoising a single chunk."""

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Post-denoising update step: appends the denoised latents to the chunk list, "
            "captures image_latents from the first chunk if needed, and extends history_latents."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return []

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor),
            InputParam("history_latents", type_hint=torch.Tensor),
            InputParam("keep_first_frame", default=True, type_hint=bool),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        # e. Collect denoised latents for this chunk
        block_state.latent_chunks.append(block_state.latents)

        # f. Update history
        if block_state.keep_first_frame and k == 0 and block_state.image_latents is None:
            block_state.image_latents = block_state.latents[:, :, 0:1, :, :]

        block_state.history_latents = torch.cat([block_state.history_latents, block_state.latents], dim=2)

        return components, block_state


# ========================================
# Chunk Loop Wrapper
# ========================================


class HeliosChunkLoopWrapper(LoopSequentialPipelineBlocks):
    """Outer chunk loop that iterates over temporal chunks.

    History indices, scheduler params, and history state are prepared by HeliosPrepareHistoryStep and
    HeliosSetTimestepsStep before this block runs. Sub-blocks handle per-chunk preparation, denoising, and history
    updates.
    """

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iterates over temporal chunks for progressive video generation. "
            "At each chunk iteration, it runs sub-blocks for preparation, denoising, and history updates."
        )

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("num_latent_chunk", required=True, type_hint=int),
        ]

    @property
    def loop_intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latent_chunks", type_hint=list, description="List of per-chunk denoised latent tensors"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.latent_chunks = []

        if not hasattr(block_state, "image_latents"):
            block_state.image_latents = None

        for k in range(block_state.num_latent_chunk):
            components, block_state = self.loop_step(components, block_state, k=k)

        self.set_block_state(state, block_state)

        return components, state


# ========================================
# Composed Chunk Denoise Steps
# ========================================


class HeliosChunkDenoiseStep(HeliosChunkLoopWrapper):
    """T2V chunk-based denoising: history slice -> noise gen -> scheduler reset -> denoise -> update."""

    block_classes = [
        HeliosChunkHistorySliceStep,
        HeliosChunkNoiseGenStep,
        HeliosChunkSchedulerResetStep,
        HeliosChunkDenoiseInner,
        HeliosChunkUpdateStep,
    ]
    block_names = ["history_slice", "noise_gen", "scheduler_reset", "denoise_inner", "update_chunk"]

    @property
    def description(self) -> str:
        return (
            "T2V chunk denoise step that iterates over temporal chunks.\n"
            "At each chunk: history_slice -> noise_gen -> scheduler_reset -> denoise_inner -> update_chunk."
        )


class HeliosI2VChunkDenoiseStep(HeliosChunkLoopWrapper):
    """I2V chunk-based denoising: I2V history slice -> noise gen -> scheduler reset -> denoise -> update."""

    block_classes = [
        HeliosI2VChunkHistorySliceStep,
        HeliosChunkNoiseGenStep,
        HeliosChunkSchedulerResetStep,
        HeliosChunkDenoiseInner,
        HeliosChunkUpdateStep,
    ]
    block_names = ["history_slice", "noise_gen", "scheduler_reset", "denoise_inner", "update_chunk"]

    @property
    def description(self) -> str:
        return (
            "I2V chunk denoise step that iterates over temporal chunks.\n"
            "At each chunk: history_slice (I2V) -> noise_gen -> scheduler_reset -> denoise_inner -> update_chunk."
        )


class HeliosPyramidDistilledChunkDenoiseInner(ModularPipelineBlocks):
    """Nested pyramid stage loop with DMD denoising for distilled checkpoints.

    Same progressive multi-resolution strategy as HeliosPyramidChunkDenoiseInner, but:
    - Guidance is disabled (guidance_scale=1.0, no unconditional pass)
    - Supports is_amplify_first_chunk (doubles first chunk's timesteps via scheduler)
    - Tracks start_point_list and passes DMD-specific args to scheduler.step()
    """

    model_name = "helios-pyramid"

    @property
    def description(self) -> str:
        return (
            "Distilled pyramid denoising inner block for DMD checkpoints. Loops over pyramid stages "
            "from smallest to full resolution with guidance disabled and DMD scheduler support."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", HeliosTransformer3DModel),
            ComponentSpec("scheduler", HeliosScheduler),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam("prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam.template("denoiser_input_fields"),
            InputParam(
                "pyramid_num_inference_steps_list",
                default=[2, 2, 2],
                type_hint=list,
                description="Number of denoising steps per pyramid stage.",
            ),
            InputParam(
                "is_amplify_first_chunk",
                default=True,
                type_hint=bool,
                description="Whether to double the first chunk's timesteps via the scheduler for amplified generation.",
            ),
            InputParam.template("attention_kwargs"),
            InputParam.template("generator"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, block_state: BlockState, k: int):
        device = components._execution_device
        transformer_dtype = components.transformer.dtype
        latents = block_state.latents
        pyramid_num_stages = len(block_state.pyramid_num_inference_steps_list)
        is_first_chunk = k == 0

        # Track start points for DMD scheduler
        start_point_list = [latents]

        # Guider inputs: only encoder_hidden_states differs between cond/uncond
        guider_inputs = {
            "encoder_hidden_states": (block_state.prompt_embeds, block_state.negative_prompt_embeds),
        }

        # Build shared kwargs from denoiser_input_fields (excludes guider-managed ones)
        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        shared_kwargs = {}
        for field_name, field_value in block_state.denoiser_input_fields.items():
            if field_name in transformer_args and field_name not in guider_inputs:
                shared_kwargs[field_name] = field_value

        # Add loop-internal history latents with dtype casting
        shared_kwargs["latents_history_short"] = block_state.latents_history_short.to(transformer_dtype)
        shared_kwargs["latents_history_mid"] = block_state.latents_history_mid.to(transformer_dtype)
        shared_kwargs["latents_history_long"] = block_state.latents_history_long.to(transformer_dtype)
        shared_kwargs["attention_kwargs"] = block_state.attention_kwargs

        for i_s in range(pyramid_num_stages):
            # --- Stage setup ---
            patch_size = components.transformer.config.patch_size

            # a. Compute mu from current resolution (before upsample, matching standard pipeline)
            image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                patch_size[0] * patch_size[1] * patch_size[2]
            )
            mu = calculate_shift(
                image_seq_len,
                components.scheduler.config.get("base_image_seq_len", 256),
                components.scheduler.config.get("max_image_seq_len", 4096),
                components.scheduler.config.get("base_shift", 0.5),
                components.scheduler.config.get("max_shift", 1.15),
            )

            # b. Set scheduler timesteps for this stage (with DMD amplification)
            num_inference_steps = block_state.pyramid_num_inference_steps_list[i_s]
            components.scheduler.set_timesteps(
                num_inference_steps,
                i_s,
                device=device,
                mu=mu,
                is_amplify_first_chunk=block_state.is_amplify_first_chunk and is_first_chunk,
            )
            timesteps = components.scheduler.timesteps

            # c. Upsample + block noise correction for stages > 0
            if i_s > 0:
                batch_size, num_channels_latents, num_frames, current_h, current_w = latents.shape
                new_h = current_h * 2
                new_w = current_w * 2

                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_frames, num_channels_latents, current_h, current_w
                )
                latents = F.interpolate(latents, size=(new_h, new_w), mode="nearest")
                latents = latents.reshape(batch_size, num_frames, num_channels_latents, new_h, new_w).permute(
                    0, 2, 1, 3, 4
                )

                # Block noise correction
                ori_sigma = 1 - components.scheduler.ori_start_sigmas[i_s]
                gamma = components.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                batch_size, num_channels_latents, num_frames, h, w = latents.shape
                noise = sample_block_noise(
                    batch_size,
                    num_channels_latents,
                    num_frames,
                    h,
                    w,
                    gamma,
                    patch_size,
                    device=device,
                    generator=block_state.generator,
                )
                noise = noise.to(dtype=transformer_dtype)
                latents = alpha * latents + beta * noise

                start_point_list.append(latents)

            # --- Timestep denoising loop ---
            num_warmup_steps = len(timesteps) - num_inference_steps * components.scheduler.order

            with tqdm(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    timestep = t.expand(latents.shape[0]).to(torch.int64)
                    latent_model_input = latents.to(transformer_dtype)

                    components.guider.set_state(step=i, num_inference_steps=num_inference_steps, timestep=t)
                    guider_state = components.guider.prepare_inputs(guider_inputs)

                    for guider_state_batch in guider_state:
                        components.guider.prepare_models(components.transformer)
                        cond_kwargs = {k: getattr(guider_state_batch, k) for k in guider_inputs.keys()}

                        context_name = getattr(guider_state_batch, components.guider._identifier_key)
                        with components.transformer.cache_context(context_name):
                            guider_state_batch.noise_pred = components.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                return_dict=False,
                                **cond_kwargs,
                                **shared_kwargs,
                            )[0]
                        components.guider.cleanup_models(components.transformer)

                    noise_pred = components.guider(guider_state)[0]

                    # Scheduler step with DMD args
                    latents = components.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        generator=block_state.generator,
                        return_dict=False,
                        cur_sampling_step=i,
                        dmd_noisy_tensor=start_point_list[i_s],
                        dmd_sigmas=components.scheduler.sigmas,
                        dmd_timesteps=components.scheduler.timesteps,
                        all_timesteps=timesteps,
                    )[0]

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % components.scheduler.order == 0
                    ):
                        progress_bar.update()

        block_state.latents = latents
        return components, block_state


class HeliosPyramidChunkDenoiseStep(HeliosChunkLoopWrapper):
    """T2V pyramid chunk denoising: history slice -> pyramid noise gen -> pyramid denoise inner -> update."""

    block_classes = [
        HeliosChunkHistorySliceStep,
        HeliosPyramidChunkNoiseGenStep,
        HeliosPyramidChunkDenoiseInner,
        HeliosChunkUpdateStep,
    ]
    block_names = ["history_slice", "noise_gen", "denoise_inner", "update_chunk"]

    @property
    def description(self) -> str:
        return (
            "T2V pyramid chunk denoise step that iterates over temporal chunks.\n"
            "At each chunk: history_slice -> noise_gen (pyramid) -> denoise_inner (pyramid stages) -> update_chunk.\n"
            "Denoising starts at the smallest resolution and progressively upsamples."
        )


class HeliosPyramidI2VChunkDenoiseStep(HeliosChunkLoopWrapper):
    """I2V pyramid chunk denoising: I2V history slice -> pyramid noise gen -> pyramid denoise inner -> update."""

    block_classes = [
        HeliosI2VChunkHistorySliceStep,
        HeliosPyramidChunkNoiseGenStep,
        HeliosPyramidChunkDenoiseInner,
        HeliosChunkUpdateStep,
    ]
    block_names = ["history_slice", "noise_gen", "denoise_inner", "update_chunk"]

    @property
    def description(self) -> str:
        return (
            "I2V pyramid chunk denoise step that iterates over temporal chunks.\n"
            "At each chunk: history_slice (I2V) -> noise_gen (pyramid) -> denoise_inner (pyramid stages) -> update_chunk.\n"
            "Denoising starts at the smallest resolution and progressively upsamples."
        )


class HeliosPyramidDistilledChunkDenoiseStep(HeliosChunkLoopWrapper):
    """T2V distilled pyramid chunk denoising with DMD scheduler and no CFG."""

    block_classes = [
        HeliosChunkHistorySliceStep,
        HeliosPyramidChunkNoiseGenStep,
        HeliosPyramidDistilledChunkDenoiseInner,
        HeliosChunkUpdateStep,
    ]
    block_names = ["history_slice", "noise_gen", "denoise_inner", "update_chunk"]

    @property
    def description(self) -> str:
        return (
            "T2V distilled pyramid chunk denoise step with DMD scheduler.\n"
            "At each chunk: history_slice -> noise_gen (pyramid) -> denoise_inner (distilled/DMD) -> update_chunk."
        )


class HeliosPyramidDistilledI2VChunkDenoiseStep(HeliosChunkLoopWrapper):
    """I2V distilled pyramid chunk denoising with DMD scheduler and no CFG."""

    block_classes = [
        HeliosI2VChunkHistorySliceStep,
        HeliosPyramidChunkNoiseGenStep,
        HeliosPyramidDistilledChunkDenoiseInner,
        HeliosChunkUpdateStep,
    ]
    block_names = ["history_slice", "noise_gen", "denoise_inner", "update_chunk"]

    @property
    def description(self) -> str:
        return (
            "I2V distilled pyramid chunk denoise step with DMD scheduler.\n"
            "At each chunk: history_slice (I2V) -> noise_gen (pyramid) -> denoise_inner (distilled/DMD) -> update_chunk."
        )
