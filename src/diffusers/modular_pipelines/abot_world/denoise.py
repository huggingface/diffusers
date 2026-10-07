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

from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from ...configuration_utils import FrozenDict
from ...models import ABotWorldTransformer3DModel, AutoencoderKLWan, AutoencoderTinyVideo
from ...models.autoencoders.autoencoder_kl_wan import WanDecodeCache
from ...models.autoencoders.autoencoder_tiny_video import TinyVideoDecodeCache
from ...models.transformers.transformer_abot_world import ABotWorldKVCache
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..modular_pipeline import IterativePipelineBlocks, ModularLoopPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ABotWorldSetActionStep(ModularLoopPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Step within the rollout loop that broadcasts this block's `[W, A, S, D, I, J, K, L]` action vector into "
            "constant pixel-resolution planes (each key repeated over 4 channels), which the transformer's action "
            "adapter encodes and adds onto the patch tokens. `action` is a loop variable: the rollout takes it from "
            "the `actions` list or callable, and a live driver passes it to "
            "`loop_step(components, state, k=k, action=action)`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", ABotWorldTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("height", type_hint=int, default=704, description="Height of the generated video in pixels"),
            InputParam("width", type_hint=int, default=1280, description="Width of the generated video in pixels"),
            InputParam(
                "num_frames_per_block",
                type_hint=int,
                default=3,
                description="Latent frames generated per block (the model was trained with 3)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "action_planes",
                type_hint=torch.Tensor,
                description="This block's broadcast action planes `[B, 32, F, height, width]`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int, action: torch.Tensor):
        block_state = self.get_block_state(state)
        device = components._execution_device

        action = torch.as_tensor(action, dtype=torch.float32)
        if action.shape != (8,):
            raise ValueError(
                f"`action` must be an 8-element `[W, A, S, D, I, J, K, L]` vector, got shape {action.shape}"
            )
        action = action.to(device=device, dtype=components.transformer.dtype)
        block_state.action_planes = (
            action.view(1, 8, 1, 1, 1)
            .repeat_interleave(4, dim=1)
            .repeat(1, 1, block_state.num_frames_per_block, block_state.height, block_state.width)
        )

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldPrepareNoiseStep(ModularLoopPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Step within the rollout loop that draws this block's initial noise and computes its token offset "
            "`current_start` in the rollout. On the first block (`current_start == 0`) the clean starting-frame "
            "latent is pinned as frame 0."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", ABotWorldTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("generator"),
            InputParam(
                "first_frame_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Normalized VAE latent of the starting frame `[B, C, 1, h, w]`",
            ),
            InputParam("height", type_hint=int, default=704, description="Height of the generated video in pixels"),
            InputParam("width", type_hint=int, default=1280, description="Width of the generated video in pixels"),
            InputParam(
                "num_frames_per_block",
                type_hint=int,
                default=3,
                description="Latent frames generated per block (the model was trained with 3)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="This block's working latents `[B, C, F, h, w]`"
            ),
            OutputParam(
                "current_start",
                type_hint=int,
                description="Token offset of this block in the rollout: `k * F * tokens_per_frame`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int, action: torch.Tensor):
        block_state = self.get_block_state(state)
        device = components._execution_device

        config = components.transformer.config
        latent_height = block_state.height // 16
        latent_width = block_state.width // 16
        frame_seqlen = (latent_height // config.patch_size[1]) * (latent_width // config.patch_size[2])
        num_frames = block_state.num_frames_per_block
        batch_size = block_state.first_frame_latents.shape[0]

        # drawn in the reference's [B, F, C, h, w] layout so a seeded run consumes the RNG identically
        noise = randn_tensor(
            (batch_size, num_frames, config.in_channels, latent_height, latent_width),
            generator=block_state.generator,
            device=device,
            dtype=components.transformer.dtype,
        )
        block_state.latents = noise.permute(0, 2, 1, 3, 4)
        block_state.current_start = k * num_frames * frame_seqlen
        if block_state.current_start == 0:
            block_state.latents[:, :, :1] = block_state.first_frame_latents.to(block_state.latents.dtype)

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldLoopDenoiser(ModularLoopPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Step within the block's denoising loop: one transformer forward at timestep `t` (per-frame; frame 0 of "
            "the first block is held at timestep 0), velocity converted to x0 with the warped sigma grid, then — "
            "except on the last step — re-noised to the next timestep with fresh noise. On the first block the "
            "clean starting-frame latent is re-pinned after every step. This block should be used to compose the "
            "`sub_blocks` attribute of an `IterativePipelineBlocks` object (e.g. `ABotWorldDenoiseLoopWrapper`)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", ABotWorldTransformer3DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="This block's working latents"),
            InputParam(
                "action_planes",
                required=True,
                type_hint=torch.Tensor,
                description="This block's broadcast action planes",
            ),
            InputParam.template("prompt_embeds"),
            InputParam(
                "reference_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Normalized VAE latents of the reference views `[B, K, C, 1, h, w]`",
            ),
            InputParam(
                "reference_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Per-slot validity mask `[B, K]` for the reference views",
            ),
            InputParam(
                "first_frame_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Normalized VAE latent of the starting frame `[B, C, 1, h, w]`",
            ),
            InputParam(
                "kv_cache",
                required=True,
                type_hint=ABotWorldKVCache,
                description="The rollout's rolling K/V cache",
            ),
            InputParam(
                "current_start",
                required=True,
                type_hint=int,
                description="Token offset of this block in the rollout",
            ),
            InputParam(
                "denoise_timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The warped denoising timesteps the loop iterates",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="The (partially) denoised block latents"),
        ]

    def _lookup_sigma(self, scheduler, timestep: torch.Tensor) -> torch.Tensor:
        """Per-element sigma via nearest-timestep lookup on the scheduler's warped grid, like the reference
        wrapper's flow -> x0 conversion (timestep 0 of the pinned first frame is off-grid, hence nearest)."""
        timesteps = scheduler.timesteps.double().to(timestep.device)
        index = torch.argmin((timesteps.unsqueeze(0) - timestep.double().unsqueeze(1)).abs(), dim=1)
        return scheduler.sigmas.double().to(timestep.device)[index]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, i: int, t: torch.Tensor):
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size, _, num_frames = block_state.latents.shape[:3]
        timestep = t.to(device).expand(batch_size, num_frames).clone()
        if block_state.current_start == 0:
            timestep[:, 0] = 0

        velocity = components.transformer(
            hidden_states=block_state.latents.to(components.transformer.dtype),
            timestep=timestep,
            encoder_hidden_states=block_state.prompt_embeds.to(components.transformer.dtype),
            action_hidden_states=block_state.action_planes,
            reference_hidden_states=block_state.reference_latents.to(components.transformer.dtype),
            reference_mask=block_state.reference_mask,
            kv_cache=block_state.kv_cache,
            current_start=block_state.current_start,
            return_dict=False,
        )[0]

        # velocity -> x0 in double precision with per-frame sigmas (frame 0 of the first block sits at
        # timestep 0), matching the reference wrapper's `_convert_flow_to_x0`
        sigma = self._lookup_sigma(components.scheduler, timestep.flatten()).reshape(batch_size, 1, num_frames, 1, 1)
        x0 = (block_state.latents.double() - sigma * velocity.double()).to(block_state.latents.dtype)

        if i < len(block_state.denoise_timesteps) - 1:
            noise = randn_tensor(
                (batch_size, num_frames, x0.shape[1], x0.shape[3], x0.shape[4]),
                generator=block_state.generator,
                device=device,
                dtype=x0.dtype,
            ).permute(0, 2, 1, 3, 4)
            next_t = block_state.denoise_timesteps[i + 1].to(device).unsqueeze(0)
            block_state.latents = components.scheduler.scale_noise(x0, next_t, noise)
        else:
            block_state.latents = x0

        if block_state.current_start == 0:
            block_state.latents[:, :, :1] = block_state.first_frame_latents.to(block_state.latents.dtype)

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldDenoiseLoopWrapper(IterativePipelineBlocks):
    model_name = "abot-world"

    @property
    def loop_variables(self) -> list[str]:
        return ["i", "t"]

    @property
    def description(self) -> str:
        return (
            "Pipeline block that denoises one rollout block over the distilled `denoise_timesteps`. It runs inside "
            "the rollout loop and accepts its loop variables (`k`, `action`)."
        )

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "denoise_timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The warped denoising timesteps the loop iterates",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int, action: torch.Tensor):
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.denoise_timesteps):
            components, state = self.loop_step(components, state, i=i, t=t)
        return components, state

    @torch.no_grad()
    def stream(self, components, state: PipelineState, k: int, action: torch.Tensor):
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.denoise_timesteps):
            components, state = yield from self.stream_step(components, state, i=i, t=t)
        return components, state


class ABotWorldDenoiseStep(ABotWorldDenoiseLoopWrapper):
    block_classes = [ABotWorldLoopDenoiser]
    block_names = ["denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step for one rollout block: the distilled few-step loop. \n"
            "Its loop logic is defined in `ABotWorldDenoiseLoopWrapper.__call__` method \n"
            "At each iteration, it runs blocks defined in `sub_blocks` sequentially:\n"
            " - `ABotWorldLoopDenoiser`\n"
        )


class ABotWorldCacheUpdateStep(ModularLoopPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Step within the rollout loop that runs one extra transformer forward on the finished block at the "
            "context noise level (timestep 0), purely to write the clean block into the K/V cache that future "
            "blocks attend over. The output is discarded."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", ABotWorldTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The denoised block latents"),
            InputParam(
                "action_planes",
                required=True,
                type_hint=torch.Tensor,
                description="This block's broadcast action planes",
            ),
            InputParam.template("prompt_embeds"),
            InputParam(
                "reference_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Normalized VAE latents of the reference views `[B, K, C, 1, h, w]`",
            ),
            InputParam(
                "reference_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Per-slot validity mask `[B, K]` for the reference views",
            ),
            InputParam(
                "kv_cache",
                required=True,
                type_hint=ABotWorldKVCache,
                description="The rollout's rolling K/V cache",
            ),
            InputParam(
                "current_start",
                required=True,
                type_hint=int,
                description="Token offset of this block in the rollout",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int, action: torch.Tensor):
        block_state = self.get_block_state(state)
        device = components._execution_device

        batch_size, _, num_frames = block_state.latents.shape[:3]
        timestep = torch.zeros((batch_size, num_frames), dtype=torch.long, device=device)

        components.transformer(
            hidden_states=block_state.latents.to(components.transformer.dtype),
            timestep=timestep,
            encoder_hidden_states=block_state.prompt_embeds.to(components.transformer.dtype),
            action_hidden_states=block_state.action_planes,
            reference_hidden_states=block_state.reference_latents.to(components.transformer.dtype),
            reference_mask=block_state.reference_mask,
            kv_cache=block_state.kv_cache,
            current_start=block_state.current_start,
            return_dict=False,
        )

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldDecodeStep(ModularLoopPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Step within the rollout loop that decodes the finished block to pixels with the full Wan VAE. The causal "
            "VAE continues from the previous blocks through a loop-carried decode cache, so only the new block's "
            "latents are decoded and the frames match a single decode of the whole rollout."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The denoised block latents"),
            InputParam(
                "decode_cache",
                type_hint=WanDecodeCache,
                description="The VAE's causal-conv cache carried over from the previous blocks; `None` for the first block",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("frames", type_hint=np.ndarray, description="This block's decoded frames `[T, H, W, 3]`"),
            OutputParam(
                "decode_cache",
                type_hint=WanDecodeCache,
                description="The VAE's causal-conv cache after this block, carried to the next block",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int, action: torch.Tensor):
        block_state = self.get_block_state(state)
        vae = components.vae

        latents = block_state.latents.to(vae.dtype)
        latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=vae.dtype).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=vae.dtype).view(1, -1, 1, 1, 1)

        cache = block_state.decode_cache if block_state.decode_cache is not None else WanDecodeCache()
        video = vae.decode(latents * latents_std + latents_mean, return_dict=False, cache=cache)[0]
        block_state.frames = components.video_processor.postprocess_video(video, output_type="np")[0]
        block_state.decode_cache = cache

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldTinyDecodeStep(ModularLoopPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Step within the rollout loop that decodes the finished block to pixels with the tiny VAE (`taew2_2`), "
            "fast enough for real-time play. The decoder continues from the previous blocks through a loop-carried "
            "decode cache, so only the new block's latents are decoded."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("tiny_vae", AutoencoderTinyVideo),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The denoised block latents"),
            InputParam(
                "decode_cache",
                type_hint=TinyVideoDecodeCache,
                description="The tiny VAE's memory carried over from the previous blocks; `None` for the first block",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("frames", type_hint=np.ndarray, description="This block's decoded frames `[T, H, W, 3]`"),
            OutputParam(
                "decode_cache",
                type_hint=TinyVideoDecodeCache,
                description="The tiny VAE's memory after this block, carried to the next block",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int, action: torch.Tensor):
        block_state = self.get_block_state(state)
        tiny_vae = components.tiny_vae

        # the tiny VAE works in the normalized latent space the transformer produces
        cache = block_state.decode_cache if block_state.decode_cache is not None else TinyVideoDecodeCache()
        video = tiny_vae.decode(block_state.latents.to(tiny_vae.dtype), return_dict=False, cache=cache)[0]
        block_state.frames = components.video_processor.postprocess_video(video, output_type="np")[0]
        block_state.decode_cache = cache

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldRolloutWrapper(IterativePipelineBlocks):
    model_name = "abot-world"

    @property
    def loop_variables(self) -> list[str]:
        return ["k", "action"]

    @property
    def description(self) -> str:
        return (
            "Pipeline block that rolls the world out block by block: at each block it encodes the block's action, "
            "draws noise, runs the distilled denoising loop against the rolling K/V cache, writes the finished block "
            "back into the cache and decodes it to pixels. `actions` is either a scripted list (one block per "
            "vector) or, with `stream()`, a callable polled once per block. Drive it through "
            "`loop_step(components, state, k=k, action=action)` to own the iteration yourself."
        )

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "actions",
                required=True,
                type_hint=list[list[int]] | Callable,
                description=(
                    "Per-block `[W, A, S, D, I, J, K, L]` 0/1 action vectors (W/A/S/D move, I/J/K/L turn the "
                    "camera): a list generates one block per vector; with `stream()`, a callable "
                    "`(block_index) -> vector or None` is polled once per block instead and the rollout runs until "
                    "it returns `None`."
                ),
            ),
        ]

    @property
    def loop_intermediate_outputs(self) -> list[OutputParam]:
        # the loop logic collects each block's decoded frames
        return [OutputParam("videos", type_hint=list[np.ndarray], description="The generated videos")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        if callable(block_state.actions):
            raise ValueError(
                "A callable `actions` needs `stream()`: with `__call__` no frames flow back between polls, so there "
                "is nothing to react to. Use `pipe.stream(...)`, or pass a scripted list instead."
            )

        frames = []
        with tqdm(total=len(block_state.actions), desc="Rollout") as progress_bar:
            for k, action in enumerate(block_state.actions):
                components, state = self.loop_step(components, state, k=k, action=action)
                frames.append(state.get("frames"))
                progress_bar.update()
        block_state.videos = [np.concatenate(frames, axis=0)]
        self.set_block_state(state, block_state)

        return components, state

    @torch.no_grad()
    def stream(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        if callable(block_state.actions):
            next_action = block_state.actions
        else:
            scripted = block_state.actions

            def next_action(k):
                return scripted[k] if k < len(scripted) else None

        frames, k = [], 0
        while (action := next_action(k)) is not None:
            components, state = yield from self.stream_step(components, state, k=k, action=action)
            frames.append(state.get("frames"))
            k += 1
        block_state.videos = [np.concatenate(frames, axis=0)]
        self.set_block_state(state, block_state)

        return components, state


class ABotWorldRolloutStep(ABotWorldRolloutWrapper):
    block_classes = [
        ABotWorldSetActionStep,
        ABotWorldPrepareNoiseStep,
        ABotWorldDenoiseStep,
        ABotWorldCacheUpdateStep,
        ABotWorldDecodeStep,
    ]
    block_names = ["set_action", "prepare_noise", "denoise", "cache_update", "decode"]

    @property
    def description(self) -> str:
        return (
            "Rollout step that generates and decodes the world block by block.\n"
            "At each block: set_action -> prepare_noise -> denoise (a nested distilled denoising loop) -> "
            "cache_update -> decode."
        )


class ABotWorldStreamingRolloutStep(ABotWorldRolloutWrapper):
    block_classes = [
        ABotWorldSetActionStep,
        ABotWorldPrepareNoiseStep,
        ABotWorldDenoiseStep,
        ABotWorldCacheUpdateStep,
        ABotWorldTinyDecodeStep,
    ]
    block_names = ["set_action", "prepare_noise", "denoise", "cache_update", "decode"]

    @property
    def description(self) -> str:
        return (
            "Streaming rollout step that generates and decodes the world block by block.\n"
            "At each block: set_action -> prepare_noise -> denoise (a nested distilled denoising loop) -> "
            "cache_update -> decode."
        )
