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

import torch

from ...guiders.magi_classifier_free_guidance import MagiClassifierFreeGuidance
from ...models import MagiTransformer3DModel
from ...schedulers import MagiEulerScheduler
from ..modular_pipeline import LoopSequentialPipelineBlocks, ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


class MagiLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Select the moving chunk window, per-chunk times, and attention ranges."

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", MagiEulerScheduler)]

    @property
    def inputs(self):
        return [
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                description="FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.",
                required=True,
            ),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Prepared conditional text features, shared across chunks or provided per chunk.",
                required=True,
            ),
            InputParam(
                "prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Boolean keep-mask matching the conditional text features.",
                required=True,
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Learned null-caption features shaped (batch, length, caption_channels), shared across chunks.",
                required=True,
            ),
            InputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Boolean keep-mask for the learned null-caption features.",
                required=True,
            ),
            InputParam(
                "prefix_chunks",
                type_hint=int,
                description="Number of supplied full-chunk prefix chunks.",
                required=True,
            ),
            InputParam(
                "num_chunks",
                type_hint=int,
                description="Total number of latent chunks, including the supplied prefix.",
                required=True,
            ),
            InputParam(
                "chunk_width", type_hint=int, description="Number of latent frames in each chunk.", required=True
            ),
            InputParam(
                "chunk_tokens",
                type_hint=int,
                description="Number of Transformer tokens per latent chunk.",
                required=True,
            ),
            InputParam(
                "steps_per_stage",
                type_hint=int,
                description="Number of iterations before the active chunk window moves forward.",
                required=True,
            ),
            InputParam(
                "window_size",
                type_hint=int,
                description="Maximum number of simultaneously denoised chunks.",
                required=True,
            ),
            InputParam(
                "num_inference_steps",
                type_hint=int,
                description="Number of Euler updates per generated chunk.",
                required=True,
            ),
            InputParam(
                "noise2clean_kvrange",
                type_hint=tuple,
                description="Positive attention-window lengths in chunks, from early to late denoising stages.",
                required=True,
            ),
            InputParam(
                "clean_chunk_kvrange",
                type_hint=int,
                description="Attention-window length used when recomputing clean chunks; `-1` uses the final noise-to-clean range.",
                required=True,
            ),
            InputParam(
                "clean_t",
                type_hint=float,
                description="Model evaluation time for clean-prefix cache extraction.",
                required=True,
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("chunk_start", type_hint=int, description="First active chunk index."),
            OutputParam("chunk_end", type_hint=int, description="Exclusive end index of the active chunk window."),
            OutputParam(
                "refresh_cache",
                type_hint=bool,
                description="Whether this iteration prepends a finalized chunk to refresh its clean KV.",
            ),
            OutputParam(
                "chunk_step_indices",
                type_hint=list,
                description="Denoising step indices for active chunks, ordered from oldest to newest.",
            ),
            OutputParam(
                "chunk_times",
                type_hint=torch.Tensor,
                description="Current per-chunk model times shaped (batch, active_chunks).",
            ),
            OutputParam(
                "next_chunk_times",
                type_hint=torch.Tensor,
                description="Next Euler endpoints shaped (batch, active_chunks).",
            ),
            OutputParam(
                "model_times",
                type_hint=torch.Tensor,
                description="Model times including an optional clean-refresh chunk.",
            ),
            OutputParam(
                "latent_model_input",
                type_hint=torch.Tensor,
                description="Current latent window including an optional clean-refresh chunk.",
            ),
            OutputParam(
                "window_prompt_embeds",
                type_hint=torch.Tensor,
                description="Conditional features for the current window, with null features for a clean-refresh chunk.",
            ),
            OutputParam(
                "window_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Boolean keep-mask matching the current window text features.",
            ),
            OutputParam(
                "kv_ranges",
                type_hint=tuple,
                description="Exclusive token attention ranges indexing the cached prefix plus the current window.",
            ),
        ]

    def __call__(self, components, block_state, i):
        stage, inner = divmod(i, block_state.steps_per_stage)
        position = block_state.prefix_chunks + stage
        block_state.chunk_start = max(block_state.prefix_chunks, position - block_state.window_size + 1)
        block_state.chunk_end = min(block_state.num_chunks, position + 1)
        t_start = max(0, position - block_state.num_chunks + 1)
        t_end = min(block_state.window_size, stage + 1)
        block_state.chunk_step_indices = [
            j * block_state.steps_per_stage + inner for j in reversed(range(t_start, t_end))
        ]
        batch = block_state.latents.shape[0]
        block_state.chunk_times = components.scheduler.timestep_schedule[block_state.chunk_step_indices][None].expand(
            batch, -1
        )
        block_state.next_chunk_times = components.scheduler.timestep_schedule[
            [j + 1 for j in block_state.chunk_step_indices]
        ][None].expand(batch, -1)
        block_state.refresh_cache = block_state.chunk_start > block_state.prefix_chunks and inner == 0
        first = block_state.chunk_start - int(block_state.refresh_cache)
        block_state.latent_model_input = block_state.latents[
            :, :, first * block_state.chunk_width : block_state.chunk_end * block_state.chunk_width
        ]
        block_state.window_prompt_embeds = block_state.prompt_embeds[
            :, block_state.chunk_start : block_state.chunk_end
        ]
        block_state.window_prompt_attention_mask = block_state.prompt_attention_mask[
            :, block_state.chunk_start : block_state.chunk_end
        ]
        block_state.model_times = block_state.chunk_times
        limits = [
            block_state.noise2clean_kvrange[
                j // (block_state.num_inference_steps // len(block_state.noise2clean_kvrange))
            ]
            for j in block_state.chunk_step_indices
        ]
        if block_state.refresh_cache:
            block_state.model_times = torch.cat(
                [block_state.latents.new_full((batch, 1), block_state.clean_t), block_state.chunk_times], dim=1
            )
            block_state.window_prompt_embeds = torch.cat(
                [block_state.negative_prompt_embeds[:, None], block_state.window_prompt_embeds], dim=1
            )
            block_state.window_prompt_attention_mask = torch.cat(
                [block_state.negative_prompt_attention_mask[:, None], block_state.window_prompt_attention_mask], dim=1
            )
            limits.insert(0, block_state.clean_chunk_kvrange)
        block_state.kv_ranges = tuple(
            (max(0, index + 1 - limit) * block_state.chunk_tokens, (index + 1) * block_state.chunk_tokens)
            for index, limit in zip(range(first, block_state.chunk_end), limits)
        )
        return components, block_state


class MagiLoopDenoiser(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Evaluate three CFG branches and retain only refreshed, unconditional clean-prefix KV."

    @property
    def expected_components(self):
        return [
            ComponentSpec("transformer", MagiTransformer3DModel),
            ComponentSpec("guider", MagiClassifierFreeGuidance),
        ]

    @property
    def inputs(self):
        return [
            InputParam(
                "latent_model_input",
                type_hint=torch.Tensor,
                description="Current latent window including an optional clean-refresh chunk.",
                required=True,
            ),
            InputParam(
                "model_times",
                type_hint=torch.Tensor,
                description="Model times including an optional clean-refresh chunk.",
                required=True,
            ),
            InputParam(
                "chunk_times",
                type_hint=torch.Tensor,
                description="Current per-chunk model times shaped (batch, active_chunks).",
                required=True,
            ),
            InputParam(
                "window_prompt_embeds",
                type_hint=torch.Tensor,
                description="Conditional features for the current window, with null features for a clean-refresh chunk.",
                required=True,
            ),
            InputParam(
                "window_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Boolean keep-mask matching the current window text features.",
                required=True,
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Learned null-caption features shaped (batch, length, caption_channels), shared across chunks.",
                required=True,
            ),
            InputParam(
                "negative_prompt_attention_mask",
                type_hint=torch.Tensor,
                description="Boolean keep-mask for the learned null-caption features.",
                required=True,
            ),
            InputParam(
                "kv_ranges",
                type_hint=tuple,
                description="Exclusive token attention ranges indexing the cached prefix plus the current window.",
                required=True,
            ),
            InputParam(
                "refresh_cache",
                type_hint=bool,
                description="Whether this iteration prepends a finalized chunk to refresh its clean KV.",
                required=True,
            ),
            InputParam("chunk_start", type_hint=int, description="First active chunk index.", required=True),
            InputParam(
                "chunk_end",
                type_hint=int,
                description="Exclusive end index of the active chunk window.",
                required=True,
            ),
            InputParam(
                "chunk_width", type_hint=int, description="Number of latent frames in each chunk.", required=True
            ),
            InputParam(
                "chunk_tokens",
                type_hint=int,
                description="Number of Transformer tokens per latent chunk.",
                required=True,
            ),
            InputParam(
                "num_inference_steps",
                type_hint=int,
                description="Number of Euler updates per generated chunk.",
                required=True,
            ),
        ] + [
            InputParam(
                "clean_kv_cache",
                type_hint=tuple,
                description="Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final generated chunk.",
                default=None,
            ),
            InputParam(
                "attention_kwargs",
                type_hint=dict,
                description="Optional keyword arguments passed to Transformer attention.",
                default=None,
            ),
            InputParam(
                "cache_device",
                type_hint=str,
                description="Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.",
                default=None,
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "velocity",
                type_hint=torch.Tensor,
                description="Three-way guided FP32 velocities for active chunks only.",
            ),
            OutputParam(
                "clean_kv_cache",
                type_hint=tuple,
                description="Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final generated chunk.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state, i):
        batch_size = block_state.latent_model_input.shape[0]
        num_chunks = block_state.chunk_end - block_state.chunk_start
        skip = int(block_state.refresh_cache) * block_state.chunk_width
        independent = (
            block_state.latent_model_input[:, :, skip:]
            .unflatten(2, (num_chunks, block_state.chunk_width))
            .transpose(1, 2)
            .flatten(0, 1)
        )
        caption_dropout_masks = torch.tensor(
            [False, True, True], device=block_state.latent_model_input.device, dtype=torch.bool
        ).split(1)
        guider = components.guider
        guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=block_state.chunk_times)
        branches = guider.prepare_inputs(
            {
                "hidden_states": (block_state.latent_model_input, block_state.latent_model_input, independent),
                "encoder_hidden_states": (
                    block_state.window_prompt_embeds,
                    block_state.negative_prompt_embeds,
                    block_state.negative_prompt_embeds.repeat_interleave(num_chunks, dim=0),
                ),
                "encoder_attention_mask": (
                    block_state.window_prompt_attention_mask,
                    block_state.negative_prompt_attention_mask,
                    block_state.negative_prompt_attention_mask.repeat_interleave(num_chunks, dim=0),
                ),
                "timestep": (block_state.model_times, block_state.model_times, block_state.chunk_times.reshape(-1, 1)),
                "caption_dropout_mask": caption_dropout_masks,
                "kv_ranges": (block_state.kv_ranges, block_state.kv_ranges, None),
                "kv_cache": (block_state.clean_kv_cache, block_state.clean_kv_cache, None),
                "use_cache": (False, block_state.refresh_cache, False),
            }
        )
        clean_kv_cache = block_state.clean_kv_cache
        for branch_index, branch in enumerate(branches):
            guider.prepare_models(components.transformer)
            try:
                result = components.transformer(
                    hidden_states=branch.hidden_states,
                    encoder_hidden_states=branch.encoder_hidden_states,
                    encoder_attention_mask=branch.encoder_attention_mask,
                    timestep=branch.timestep,
                    caption_dropout_mask=branch.caption_dropout_mask,
                    kv_ranges=branch.kv_ranges,
                    kv_cache=branch.kv_cache,
                    use_cache=branch.use_cache,
                    cache_token_count=block_state.chunk_start * block_state.chunk_tokens if branch.use_cache else None,
                    cache_device=block_state.cache_device if branch.use_cache else None,
                    attention_kwargs=block_state.attention_kwargs,
                )
            finally:
                guider.cleanup_models(components.transformer)
            if branch_index < 2:
                branch.noise_pred = result.sample[:, :, skip:]
            else:
                branch.noise_pred = result.sample.unflatten(0, (batch_size, num_chunks)).transpose(1, 2).flatten(2, 3)
            if branch.use_cache:
                clean_kv_cache = result.kv_cache
            del result
        block_state.velocity = guider(branches).pred
        block_state.clean_kv_cache = clean_kv_cache
        return components, block_state


class MagiLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Integrate active chunks in FP32 without changing finalized prefix latents."

    @property
    def expected_components(self):
        return [ComponentSpec("scheduler", MagiEulerScheduler)]

    @property
    def inputs(self):
        return [
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                description="FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.",
                required=True,
            ),
            InputParam(
                "velocity",
                type_hint=torch.Tensor,
                description="Three-way guided FP32 velocities for active chunks only.",
                required=True,
            ),
            InputParam(
                "chunk_times",
                type_hint=torch.Tensor,
                description="Current per-chunk model times shaped (batch, active_chunks).",
                required=True,
            ),
            InputParam(
                "next_chunk_times",
                type_hint=torch.Tensor,
                description="Next Euler endpoints shaped (batch, active_chunks).",
                required=True,
            ),
            InputParam("chunk_start", type_hint=int, description="First active chunk index.", required=True),
            InputParam(
                "chunk_end",
                type_hint=int,
                description="Exclusive end index of the active chunk window.",
                required=True,
            ),
            InputParam(
                "chunk_width", type_hint=int, description="Number of latent frames in each chunk.", required=True
            ),
            InputParam(
                "chunk_step_indices",
                type_hint=list,
                description="Denoising step indices for active chunks, ordered from oldest to newest.",
                required=True,
            ),
            InputParam(
                "num_inference_steps",
                type_hint=int,
                description="Number of Euler updates per generated chunk.",
                required=True,
            ),
            InputParam(
                "completed_chunks",
                type_hint=list,
                description="Indices of supplied prefix chunks and finalized generated chunks.",
                required=True,
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.",
            ),
            OutputParam(
                "completed_chunks",
                type_hint=list,
                description="Indices of supplied prefix chunks and finalized generated chunks.",
            ),
        ]

    def __call__(self, components, block_state, i):
        start, end = block_state.chunk_start * block_state.chunk_width, block_state.chunk_end * block_state.chunk_width
        block_state.latents[:, :, start:end] = components.scheduler.step(
            block_state.velocity,
            block_state.chunk_times,
            block_state.latents[:, :, start:end],
            next_timestep=block_state.next_chunk_times,
        ).prev_sample
        if block_state.chunk_step_indices[0] == block_state.num_inference_steps - 1:
            block_state.completed_chunks.append(block_state.chunk_start)
        return components, block_state


# auto_docstring
class MagiDenoiseLoop(LoopSequentialPipelineBlocks):
    """
    Run the asynchronous MAGI chunk-denoising window with a clean-prefix cache.

      Components:
          - scheduler (`MagiEulerScheduler`)
          - transformer (`MagiTransformer3DModel`)
          - guider (`MagiClassifierFreeGuidance`)

      Inputs:
          num_window_steps (`int`):
              Total number of asynchronous window iterations.
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
          prompt_embeds (`Tensor`):
              Prepared conditional text features, shared across chunks or provided per chunk.
          prompt_attention_mask (`Tensor`):
              Boolean keep-mask matching the conditional text features.
          negative_prompt_embeds (`Tensor`):
              Learned null-caption features shaped (batch, length, caption_channels), shared across chunks.
          negative_prompt_attention_mask (`Tensor`):
              Boolean keep-mask for the learned null-caption features.
          prefix_chunks (`int`):
              Number of supplied full-chunk prefix chunks.
          num_chunks (`int`):
              Total number of latent chunks, including the supplied prefix.
          chunk_width (`int`):
              Number of latent frames in each chunk.
          chunk_tokens (`int`):
              Number of Transformer tokens per latent chunk.
          steps_per_stage (`int`):
              Number of iterations before the active chunk window moves forward.
          window_size (`int`):
              Maximum number of simultaneously denoised chunks.
          num_inference_steps (`int`):
              Number of Euler updates per generated chunk.
          noise2clean_kvrange (`tuple`):
              Positive attention-window lengths in chunks, from early to late denoising stages.
          clean_chunk_kvrange (`int`):
              Attention-window length used when recomputing clean chunks; `-1` uses the final noise-to-clean range.
          clean_t (`float`):
              Model evaluation time for clean-prefix cache extraction.
          clean_kv_cache (`tuple`, *optional*):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final generated chunk.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.

      Outputs:
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
    """

    model_name = "magi"
    block_classes = [MagiLoopBeforeDenoiser, MagiLoopDenoiser, MagiLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self):
        return "Run the asynchronous MAGI chunk-denoising window with a clean-prefix cache."

    @property
    def loop_inputs(self):
        return [
            InputParam(
                "num_window_steps",
                type_hint=int,
                description="Total number of asynchronous window iterations.",
                required=True,
            )
        ]

    @property
    def outputs(self):
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        components.scheduler.set_timesteps(block_state.num_inference_steps, device=block_state.latents.device)
        with self.progress_bar(total=block_state.num_window_steps) as progress:
            for i in range(block_state.num_window_steps):
                components, block_state = self.loop_step(components, block_state, i=i)
                progress.update()
        self.set_block_state(state, block_state)
        return components, state


class MagiPrefixLoopBeforeDenoiser(MagiLoopBeforeDenoiser):
    @property
    def inputs(self):
        return super().inputs + [
            InputParam(
                "conditioning_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Original scaled prefix, reinjected before every model evaluation.",
            )
        ]

    @property
    def intermediate_outputs(self):
        return super().intermediate_outputs + [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.",
            )
        ]

    def __call__(self, components, block_state, i):
        components, block_state = super().__call__(components, block_state, i)
        first = (block_state.chunk_start - int(block_state.refresh_cache)) * block_state.chunk_width
        start = block_state.chunk_start * block_state.chunk_width
        end = min(block_state.conditioning_latents.shape[2], block_state.chunk_end * block_state.chunk_width)
        # Euler integrates from the injected prefix, but cache refresh must not overwrite finalized output.
        if start < end:
            block_state.latents[:, :, start:end] = block_state.conditioning_latents[:, :, start:end]
        if block_state.refresh_cache and first < end:
            block_state.latent_model_input = block_state.latent_model_input.clone()
            block_state.latent_model_input[:, :, : end - first] = block_state.conditioning_latents[:, :, first:end]
        return components, block_state


# auto_docstring
class MagiPrefixDenoiseLoop(MagiDenoiseLoop):
    """
    Denoise with per-evaluation prefix injection and separate clean-prefix cache refresh.

      Components:
          - scheduler (`MagiEulerScheduler`)
          - transformer (`MagiTransformer3DModel`)
          - guider (`MagiClassifierFreeGuidance`)

      Inputs:
          num_window_steps (`int`):
              Total number of asynchronous window iterations.
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
          prompt_embeds (`Tensor`):
              Prepared conditional text features, shared across chunks or provided per chunk.
          prompt_attention_mask (`Tensor`):
              Boolean keep-mask matching the conditional text features.
          negative_prompt_embeds (`Tensor`):
              Learned null-caption features shaped (batch, length, caption_channels), shared across chunks.
          negative_prompt_attention_mask (`Tensor`):
              Boolean keep-mask for the learned null-caption features.
          prefix_chunks (`int`):
              Number of supplied full-chunk prefix chunks.
          num_chunks (`int`):
              Total number of latent chunks, including the supplied prefix.
          chunk_width (`int`):
              Number of latent frames in each chunk.
          chunk_tokens (`int`):
              Number of Transformer tokens per latent chunk.
          steps_per_stage (`int`):
              Number of iterations before the active chunk window moves forward.
          window_size (`int`):
              Maximum number of simultaneously denoised chunks.
          num_inference_steps (`int`):
              Number of Euler updates per generated chunk.
          noise2clean_kvrange (`tuple`):
              Positive attention-window lengths in chunks, from early to late denoising stages.
          clean_chunk_kvrange (`int`):
              Attention-window length used when recomputing clean chunks; `-1` uses the final noise-to-clean range.
          clean_t (`float`):
              Model evaluation time for clean-prefix cache extraction.
          conditioning_latents (`Tensor`):
              Original scaled prefix, reinjected before every model evaluation.
          clean_kv_cache (`tuple`, *optional*):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final generated chunk.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.

      Outputs:
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
    """

    model_name = "magi"
    block_classes = [MagiPrefixLoopBeforeDenoiser, MagiLoopDenoiser, MagiLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self):
        return "Denoise with per-evaluation prefix injection and separate clean-prefix cache refresh."
