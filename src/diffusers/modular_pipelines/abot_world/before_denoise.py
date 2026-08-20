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

from ...models import ABotWorldTransformer3DModel
from ...models.transformers.transformer_abot_world import ABotWorldKVCache
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ABotWorldPrepareStep(ModularPipelineBlocks):
    model_name = "abot-world"

    @property
    def description(self) -> str:
        return (
            "Prepare step for the causal rollout: sets the scheduler's full shifted flow-match grid and warps the "
            "distilled `denoising_timesteps` through it, validates the per-block actions, and allocates the "
            "transformer's rolling K/V cache with the reference tokens pinned at its head."
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
            InputParam(
                "actions",
                required=True,
                type_hint=list[list[int]],
                description=(
                    "Per-block actions, one `[W, A, S, D, I, J, K, L]` 0/1 vector per generated block "
                    "(W/A/S/D move, I/J/K/L turn the camera). The rollout generates `len(actions)` blocks."
                ),
            ),
            InputParam(
                "denoising_timesteps",
                type_hint=list[int],
                default=[1000, 750, 500, 250],
                description="The distilled student's denoising timesteps, before shift-warping",
            ),
            InputParam("height", type_hint=int, default=704, description="Height of the generated video in pixels"),
            InputParam("width", type_hint=int, default=1280, description="Width of the generated video in pixels"),
            InputParam(
                "reference_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Normalized VAE latents of the reference views `[B, K, C, 1, h, w]`",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("actions", type_hint=torch.Tensor, description="The actions as a `[num_blocks, 8]` tensor"),
            OutputParam(
                "denoise_timesteps",
                type_hint=torch.Tensor,
                description="The warped denoising timesteps the rollout loop iterates",
            ),
            OutputParam("kv_cache", type_hint=ABotWorldKVCache, description="The rollout's rolling K/V cache"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.actions = torch.tensor(block_state.actions, dtype=torch.float32)
        if block_state.actions.ndim != 2 or block_state.actions.shape[1] != 8:
            raise ValueError(f"`actions` must be a list of 8-element vectors, got shape {block_state.actions.shape}")

        # the full 1000-point flow-match grid the reference warps its step list through: the scheduler
        # applies its configured shift to sigmas = linspace(1, 0, 1001)[:-1]
        components.scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.0, 1001)[:-1].tolist())
        timesteps = components.scheduler.timesteps.float()
        step_list = torch.tensor(block_state.denoising_timesteps, dtype=torch.long)
        block_state.denoise_timesteps = torch.cat([timesteps, timesteps.new_zeros(1)])[1000 - step_list]

        config = components.transformer.config
        frame_seqlen = (block_state.height // 16 // config.patch_size[1]) * (
            block_state.width // 16 // config.patch_size[2]
        )
        num_slots, _, ref_t, ref_h, ref_w = block_state.reference_latents.shape[1:]
        ref_token_len = (
            num_slots
            * (ref_t // config.patch_size[0])
            * (ref_h // config.patch_size[1])
            * (ref_w // config.patch_size[2])
        )
        block_state.kv_cache = ABotWorldKVCache(
            num_layers=config.num_layers,
            batch_size=block_state.reference_latents.shape[0],
            num_tokens=ref_token_len + config.local_attn_size * frame_seqlen,
            ref_token_len=ref_token_len,
            num_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
            device=device,
            dtype=components.transformer.dtype,
        )

        self.set_block_state(state, block_state)
        return components, state
