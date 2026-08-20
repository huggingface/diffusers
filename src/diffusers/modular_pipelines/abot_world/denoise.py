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
from ...models import ABotWorldTransformer3DModel, AutoencoderKLWan
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
            "adapter encodes and adds onto the patch tokens. An interactive driver overwrites `actions` in the "
            "state between `loop_step` calls to steer the world live."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", ABotWorldTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "actions",
                required=True,
                type_hint=torch.Tensor,
                description="Per-block action vectors `[num_blocks, 8]`, from the prepare step",
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
                "action_planes",
                type_hint=torch.Tensor,
                description="This block's broadcast action planes `[B, 32, F, height, width]`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int):
        block_state = self.get_block_state(state)
        device = components._execution_device

        action = block_state.actions[k].to(device=device, dtype=components.transformer.dtype)
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
    def __call__(self, components, state: PipelineState, k: int):
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
            "the rollout loop and reads the current block index `k` from the loop scope."
        )

    @property
    def inputs(self) -> list[InputParam]:
        inputs = super().inputs
        names = {param.name for param in inputs}
        # inputs consumed by the loop logic itself, on top of what the sub-blocks declare
        loop_inputs = [
            InputParam(
                "denoise_timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The warped denoising timesteps the loop iterates",
            ),
        ]
        return [param for param in loop_inputs if param.name not in names] + inputs

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int):
        block_state = self.get_block_state(state)
        for i, t in enumerate(block_state.denoise_timesteps):
            components, state = self.loop_step(components, state, i=i, t=t)
        return components, state

    @torch.no_grad()
    def stream(self, components, state: PipelineState, k: int):
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
    def __call__(self, components, state: PipelineState, k: int):
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
            kv_cache=block_state.kv_cache,
            current_start=block_state.current_start,
            return_dict=False,
        )

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldRolloutWrapper(IterativePipelineBlocks):
    model_name = "abot-world"

    @property
    def loop_variables(self) -> list[str]:
        return ["k"]

    @property
    def description(self) -> str:
        return (
            "Pipeline block that rolls the world out block by block: at each block it encodes the block's action, "
            "draws noise, runs the distilled denoising loop against the rolling K/V cache, and writes the finished "
            "block back into the cache. Drive it through `loop_step(components, state, k=k)` to own the iteration — "
            "write new `actions` into the state between calls for live interaction."
        )

    @property
    def inputs(self) -> list[InputParam]:
        inputs = super().inputs
        names = {param.name for param in inputs}
        # `actions` is also consumed by the loop logic itself (the rollout length)
        loop_inputs = [
            InputParam(
                "actions",
                required=True,
                type_hint=torch.Tensor,
                description="Per-block action vectors `[num_blocks, 8]`, from the prepare step",
            ),
        ]
        return [param for param in loop_inputs if param.name not in names] + inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        # produced by the loop logic itself, which collects each finished block
        return super().intermediate_outputs + [
            OutputParam(
                "video_latents",
                type_hint=torch.Tensor,
                description="The rollout's accumulated latents `[B, C, num_blocks * F, h, w]`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        if block_state.actions is None:
            raise ValueError(
                "A scripted rollout requires `actions` (one vector per block); to drive the rollout "
                "interactively, call `loop_step` yourself and write `action` into the state between calls."
            )

        video_latents = []
        with tqdm(total=block_state.actions.shape[0], desc="Rollout") as progress_bar:
            for k in range(block_state.actions.shape[0]):
                components, state = self.loop_step(components, state, k=k)
                video_latents.append(state.get("latents"))
                progress_bar.update()
        state.set("video_latents", torch.cat(video_latents, dim=2))

        return components, state

    @torch.no_grad()
    def stream(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        if block_state.actions is None:
            raise ValueError(
                "A scripted rollout requires `actions` (one vector per block); to drive the rollout "
                "interactively, call `loop_step` yourself and write `action` into the state between calls."
            )

        video_latents = []
        for k in range(block_state.actions.shape[0]):
            components, state = yield from self.stream_step(components, state, k=k)
            video_latents.append(state.get("latents"))
        state.set("video_latents", torch.cat(video_latents, dim=2))

        return components, state


class ABotWorldRolloutStep(ABotWorldRolloutWrapper):
    block_classes = [
        ABotWorldSetActionStep,
        ABotWorldPrepareNoiseStep,
        ABotWorldDenoiseStep,
        ABotWorldCacheUpdateStep,
    ]
    block_names = ["set_action", "prepare_noise", "denoise", "cache_update"]

    @property
    def description(self) -> str:
        return (
            "Rollout step that generates the world block by block.\n"
            "At each block: set_action -> prepare_noise -> denoise (a nested distilled denoising loop) -> "
            "cache_update."
        )


class ABotWorldCurrentActionStep(ModularLoopPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Step within the streaming rollout loop that broadcasts the *current* `[W, A, S, D, I, J, K, L]` action "
            "vector (the `action` state value) into the conditioning planes. The scripted `__call__`/`stream` set "
            "`action` from the `actions` list each iteration; a live driver writes it into the state between "
            "`loop_step` calls."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", ABotWorldTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "action",
                required=True,
                type_hint=torch.Tensor,
                description="The current block's `[W, A, S, D, I, J, K, L]` 0/1 action vector",
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
                "action_planes",
                type_hint=torch.Tensor,
                description="This block's broadcast action planes `[B, 32, F, height, width]`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int):
        block_state = self.get_block_state(state)
        device = components._execution_device

        action = torch.as_tensor(block_state.action, dtype=torch.float32)
        action = action.to(device=device, dtype=components.transformer.dtype)
        block_state.action_planes = (
            action.view(1, 8, 1, 1, 1)
            .repeat_interleave(4, dim=1)
            .repeat(1, 1, block_state.num_frames_per_block, block_state.height, block_state.width)
        )

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldStreamingDecodeStep(ModularLoopPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Step within the streaming rollout loop that decodes the finished block to pixels. The causal VAE needs "
            "temporal context, so the previous block's latents are decoded alongside and only the new block's "
            "frames are kept."
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
                "previous_latents",
                type_hint=torch.Tensor,
                description="The previous block's latents, decoded as temporal context; `None` for the first block",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("frames", type_hint=np.ndarray, description="This block's decoded frames `[T, H, W, 3]`"),
            OutputParam(
                "previous_latents",
                type_hint=torch.Tensor,
                description="This block's latents, kept as the next block's decode context",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState, k: int):
        block_state = self.get_block_state(state)
        vae = components.vae

        latents = block_state.latents
        if block_state.previous_latents is None:
            decode_input = latents
            new_frames = None  # keep everything
        else:
            decode_input = torch.cat([block_state.previous_latents, latents], dim=2)
            new_frames = latents.shape[2] * 4

        decode_input = decode_input.to(vae.dtype)
        latents_mean = torch.tensor(vae.config.latents_mean, device=decode_input.device, dtype=vae.dtype).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(vae.config.latents_std, device=decode_input.device, dtype=vae.dtype).view(
            1, -1, 1, 1, 1
        )
        video = vae.decode(decode_input * latents_std + latents_mean, return_dict=False)[0]
        video = components.video_processor.postprocess_video(video, output_type="np")[0]

        block_state.frames = video if new_frames is None else video[-new_frames:]
        block_state.previous_latents = latents

        self.set_block_state(state, block_state)
        return components, state


class ABotWorldStreamingRolloutWrapper(IterativePipelineBlocks):
    model_name = "abot-world"

    @property
    def loop_variables(self) -> list[str]:
        return ["k"]

    @property
    def description(self) -> str:
        return (
            "Pipeline block that rolls the world out block by block and decodes each block to pixels inside the "
            "loop. Scripted runs iterate the `actions` list (each iteration writes the block's `action` into the "
            "state, exactly as a live driver would); interactive drivers own the loop via "
            "`loop_step(components, state, k=k)` and write `action` between calls."
        )

    @property
    def inputs(self) -> list[InputParam]:
        # `action` and `previous_latents` are loop-carried — supplied per iteration by the loop logic (or a live
        # driver) and by the decode step of the previous iteration — never user-provided.
        inputs = [param for param in super().inputs if param.name not in ("action", "previous_latents")]
        names = {param.name for param in inputs}
        # inputs consumed by the loop logic itself
        loop_inputs = [
            InputParam(
                "actions",
                type_hint=torch.Tensor,
                description="Per-block action vectors `[num_blocks, 8]`, from the prepare step",
            ),
            InputParam(
                "action_source",
                type_hint=Callable,
                description=(
                    "Interactive alternative to `actions`: a callable `(block_index) -> action vector or None` "
                    "polled once per block — return the current `[W, A, S, D, I, J, K, L]` input to keep rolling, "
                    "or `None` to stop. The rollout is unbounded while it returns actions."
                ),
            ),
        ]
        return [param for param in loop_inputs if param.name not in names] + inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        # produced by the loop logic itself, which collects each block's decoded frames
        return super().intermediate_outputs + [
            OutputParam("videos", type_hint=list[np.ndarray], description="The generated videos"),
        ]

    def _next_action(self, block_state, k):
        if block_state.actions is not None:
            return block_state.actions[k] if k < block_state.actions.shape[0] else None
        if block_state.action_source is not None:
            action = block_state.action_source(k)
            return None if action is None else torch.as_tensor(action, dtype=torch.float32)
        raise ValueError(
            "The streaming rollout needs `actions` (a scripted list) or `action_source` (a callable polled per "
            "block); to own the loop yourself instead, call `loop_step` and write `action` into the state "
            "between calls."
        )

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        if block_state.actions is None and block_state.action_source is not None:
            raise ValueError(
                "`action_source` needs `stream()`: with `__call__` no frames flow back between polls, so there "
                "is nothing to react to. Use `pipe.stream(...)`, or pass a scripted `actions` list instead."
            )

        frames, k = [], 0
        while (action := self._next_action(block_state, k)) is not None:
            state.set("action", action)
            components, state = self.loop_step(components, state, k=k)
            frames.append(state.get("frames"))
            k += 1
        state.set("videos", [np.concatenate(frames, axis=0)])

        return components, state

    @torch.no_grad()
    def stream(self, components, state: PipelineState):
        block_state = self.get_block_state(state)

        frames, k = [], 0
        while (action := self._next_action(block_state, k)) is not None:
            state.set("action", action)
            components, state = yield from self.stream_step(components, state, k=k)
            frames.append(state.get("frames"))
            k += 1
        state.set("videos", [np.concatenate(frames, axis=0)])

        return components, state


class ABotWorldStreamingRolloutStep(ABotWorldStreamingRolloutWrapper):
    block_classes = [
        ABotWorldCurrentActionStep,
        ABotWorldPrepareNoiseStep,
        ABotWorldDenoiseStep,
        ABotWorldCacheUpdateStep,
        ABotWorldStreamingDecodeStep,
    ]
    block_names = ["set_action", "prepare_noise", "denoise", "cache_update", "decode"]

    @property
    def description(self) -> str:
        return (
            "Streaming rollout step that generates and decodes the world block by block.\n"
            "At each block: set_action -> prepare_noise -> denoise (a nested distilled denoising loop) -> "
            "cache_update -> decode."
        )
