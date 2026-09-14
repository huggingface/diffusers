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
from ..modular_pipeline import LoopSequentialPipelineBlocks, ModularPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


_STATE_FIELDS = {
    "latents": (
        torch.Tensor,
        "FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.",
    ),
    "prompt_embeds": (torch.Tensor, "Prepared conditional text features, shared across chunks or provided per chunk."),
    "prompt_attention_mask": (torch.Tensor, "Boolean keep-mask matching the conditional text features."),
    "negative_prompt_embeds": (
        torch.Tensor,
        "Learned null-caption features shaped (batch, length, caption_channels), shared across chunks.",
    ),
    "negative_prompt_attention_mask": (torch.Tensor, "Boolean keep-mask for the learned null-caption features."),
    "prefix_latents": (
        torch.Tensor,
        "Optional full-chunk clean prefix; replaces the leading latent slots and remains unchanged.",
    ),
    "num_inference_steps": (int, "Number of Euler updates per generated chunk."),
    "chunk_width": (int, "Number of latent frames in each chunk."),
    "window_size": (int, "Maximum number of simultaneously denoised chunks."),
    "noise2clean_kvrange": (
        tuple,
        "Positive attention-window lengths in chunks, from early to late denoising stages.",
    ),
    "clean_chunk_kvrange": (int, "Positive attention-window length used when recomputing clean chunks."),
    "clean_t": (float, "Model evaluation time for clean-prefix cache extraction."),
    "cache_device": (
        str,
        "Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.",
    ),
    "attention_kwargs": (dict, "Optional keyword arguments passed to Transformer attention."),
    "num_chunks": (int, "Total number of latent chunks, including the supplied prefix."),
    "prefix_chunks": (int, "Number of supplied full-chunk prefix chunks."),
    "chunk_tokens": (int, "Number of Transformer tokens per latent chunk."),
    "steps_per_stage": (int, "Number of iterations before the active chunk window moves forward."),
    "num_window_steps": (int, "Total number of asynchronous window iterations."),
    "timestep_schedule": (torch.Tensor, "FP32 schedule including the final Euler integration endpoint."),
    "clean_kv_cache": (
        tuple,
        "Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final generated chunk.",
    ),
    "completed_chunks": (list, "Indices of supplied prefix chunks and finalized generated chunks."),
    "chunk_start": (int, "First active chunk index."),
    "chunk_end": (int, "Exclusive end index of the active chunk window."),
    "refresh_cache": (bool, "Whether this iteration prepends a finalized chunk to refresh its clean KV."),
    "chunk_step_indices": (list, "Denoising step indices for active chunks, ordered from oldest to newest."),
    "chunk_times": (torch.Tensor, "Current per-chunk model times shaped (batch, active_chunks)."),
    "next_chunk_times": (torch.Tensor, "Next Euler endpoints shaped (batch, active_chunks)."),
    "model_times": (torch.Tensor, "Model times including an optional clean-refresh chunk."),
    "latent_model_input": (torch.Tensor, "Current latent window including an optional clean-refresh chunk."),
    "window_prompt_embeds": (
        torch.Tensor,
        "Conditional features for the current window, with null features for a clean-refresh chunk.",
    ),
    "window_prompt_attention_mask": (torch.Tensor, "Boolean keep-mask matching the current window text features."),
    "kv_ranges": (tuple, "Exclusive token attention ranges indexing the cached prefix plus the current window."),
    "velocity": (torch.Tensor, "Three-way guided FP32 velocities for active chunks only."),
}


def _input(name, **kwargs):
    type_hint, description = _STATE_FIELDS[name]
    return InputParam(name, type_hint=type_hint, description=description, **kwargs)


def _output(name):
    type_hint, description = _STATE_FIELDS[name]
    return OutputParam(name, type_hint=type_hint, description=description)


def _ranges(chunk_indices, limits, chunk_tokens):
    return tuple(
        (max(0, index + 1 - limit) * chunk_tokens, (index + 1) * chunk_tokens)
        for index, limit in zip(chunk_indices, limits)
    )


class MagiPrepareDenoiseStep(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Prepare the MAGI base-model chunk schedule and optional full-chunk clean prefix."

    @property
    def expected_components(self):
        return [ComponentSpec("transformer", MagiTransformer3DModel), ComponentSpec("scheduler", MagiEulerScheduler)]

    @property
    def inputs(self):
        return [
            _input("latents", required=True),
            _input("prompt_embeds", required=True),
            _input("prompt_attention_mask", required=True),
            _input("negative_prompt_embeds", required=True),
            _input("negative_prompt_attention_mask", required=True),
            _input("prefix_latents", default=None),
            _input("num_inference_steps", default=64),
            _input("chunk_width", default=6),
            _input("window_size", default=4),
            _input("noise2clean_kvrange", default=(5, 4, 3, 2)),
            _input("clean_chunk_kvrange", default=1),
            _input("clean_t", default=0.9999),
            _input("attention_kwargs", default=None),
            _input("cache_device", default=None),
        ]

    @property
    def intermediate_outputs(self):
        return [
            _output(name)
            for name in [
                "num_chunks",
                "prefix_chunks",
                "chunk_tokens",
                "steps_per_stage",
                "num_window_steps",
                "timestep_schedule",
                "clean_kv_cache",
                "completed_chunks",
            ]
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        s = self.get_block_state(state)
        config = components.transformer.config
        if config.distilled:
            raise ValueError("MagiDenoiseStep supports base models only, not distilled models.")
        for name in ("chunk_width", "window_size", "num_inference_steps", "clean_chunk_kvrange"):
            value = getattr(s, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not s.noise2clean_kvrange or any(not isinstance(x, int) or x <= 0 for x in s.noise2clean_kvrange):
            raise ValueError("noise2clean_kvrange must contain positive chunk counts.")
        if s.num_inference_steps % s.window_size or s.num_inference_steps % len(s.noise2clean_kvrange):
            raise ValueError("num_inference_steps must be divisible by window_size and the number of KV ranges.")
        if not 0 <= s.clean_t <= 1:
            raise ValueError("clean_t must be in [0, 1].")
        if s.latents.ndim != 5 or min(s.latents.shape) <= 0 or s.latents.shape[1] != config.in_channels:
            raise ValueError("latents must have shape (batch, in_channels, frames, height, width).")
        batch, channels, frames, height, width = s.latents.shape
        pt, ph, pw = config.patch_size
        if frames % s.chunk_width or s.chunk_width % pt or height % ph or width % pw:
            raise ValueError("Latents must contain full chunks and be divisible by the Transformer patch size.")
        s.num_chunks = frames // s.chunk_width
        for name, mask_name, per_chunk in [
            ("prompt_embeds", "prompt_attention_mask", True),
            ("negative_prompt_embeds", "negative_prompt_attention_mask", False),
        ]:
            features, mask = getattr(s, name), getattr(s, mask_name)
            valid_shape = (features.ndim == 3 and features.shape[0] == batch) or (
                per_chunk and features.ndim == 4 and features.shape[:2] == (batch, s.num_chunks)
            )
            if not valid_shape or features.shape[-1] != config.caption_channels or features.shape[-2] <= 0:
                raise ValueError(f"{name} must match the batch, caption channels, and optional chunk count.")
            if mask.shape != features.shape[:-1] or mask.dtype != torch.bool or not mask.any(dim=-1).all():
                raise ValueError(f"{mask_name} must be a boolean keep-mask with at least one valid token per caption.")
            if features.device != s.latents.device or mask.device != s.latents.device:
                raise ValueError("All latents, text features, and masks must be on the same device.")
        if s.prompt_embeds.shape[-2] != s.negative_prompt_embeds.shape[-2]:
            raise ValueError("Conditional and null text features must have the same padded length.")
        s.latents = s.latents.float().clone()
        if s.prompt_embeds.ndim == 3:
            s.prompt_embeds = s.prompt_embeds[:, None].expand(-1, s.num_chunks, -1, -1)
            s.prompt_attention_mask = s.prompt_attention_mask[:, None].expand(-1, s.num_chunks, -1)
        s.prefix_chunks = 0
        s.clean_kv_cache = None
        s.chunk_tokens = (s.chunk_width // pt) * (height // ph) * (width // pw)
        if s.prefix_latents is not None:
            prefix = s.prefix_latents
            if (
                prefix.ndim != 5
                or prefix.shape[:2] != (batch, channels)
                or prefix.shape[3:] != (height, width)
                or prefix.device != s.latents.device
            ):
                raise ValueError(
                    "prefix_latents must match the latent batch, channels, spatial dimensions, and device."
                )
            if prefix.shape[2] <= 0 or prefix.shape[2] % s.chunk_width or prefix.shape[2] >= frames:
                raise ValueError("prefix_latents must contain full chunks and leave at least one chunk to generate.")
            s.prefix_chunks = prefix.shape[2] // s.chunk_width
            s.latents[:, :, : prefix.shape[2]] = prefix.float()
            s.clean_kv_cache = components.transformer(
                hidden_states=s.latents[:, :, : prefix.shape[2]],
                encoder_hidden_states=s.negative_prompt_embeds,
                encoder_attention_mask=s.negative_prompt_attention_mask,
                caption_dropout_mask=torch.ones(1, device=s.latents.device, dtype=torch.bool),
                timestep=s.latents.new_full((batch, s.prefix_chunks), s.clean_t),
                kv_ranges=_ranges(range(s.prefix_chunks), [s.clean_chunk_kvrange] * s.prefix_chunks, s.chunk_tokens),
                use_cache=True,
                cache_device=s.cache_device,
                attention_kwargs=s.attention_kwargs,
            ).kv_cache
        s.steps_per_stage = s.num_inference_steps // s.window_size
        s.num_window_steps = s.steps_per_stage * (s.num_chunks + s.window_size - 1 - s.prefix_chunks)
        s.completed_chunks = list(range(s.prefix_chunks))
        components.scheduler.set_timesteps(s.num_inference_steps, device=s.latents.device)
        s.timestep_schedule = components.scheduler.timestep_schedule
        self.set_block_state(state, s)
        return components, state


class MagiLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Select the moving chunk window, per-chunk times, and attention ranges."

    @property
    def inputs(self):
        return [
            _input(name, required=True)
            for name in [
                "latents",
                "prompt_embeds",
                "prompt_attention_mask",
                "negative_prompt_embeds",
                "negative_prompt_attention_mask",
                "prefix_chunks",
                "num_chunks",
                "chunk_width",
                "chunk_tokens",
                "steps_per_stage",
                "window_size",
                "num_inference_steps",
                "noise2clean_kvrange",
                "clean_chunk_kvrange",
                "clean_t",
                "timestep_schedule",
            ]
        ]

    @property
    def intermediate_outputs(self):
        return [
            _output(name)
            for name in [
                "chunk_start",
                "chunk_end",
                "refresh_cache",
                "chunk_step_indices",
                "chunk_times",
                "next_chunk_times",
                "model_times",
                "latent_model_input",
                "window_prompt_embeds",
                "window_prompt_attention_mask",
                "kv_ranges",
            ]
        ]

    def __call__(self, components, s, i):
        stage, inner = divmod(i, s.steps_per_stage)
        position = s.prefix_chunks + stage
        s.chunk_start = max(s.prefix_chunks, position - s.window_size + 1)
        s.chunk_end = min(s.num_chunks, position + 1)
        t_start = max(0, position - s.num_chunks + 1)
        t_end = min(s.window_size, stage + 1)
        s.chunk_step_indices = [j * s.steps_per_stage + inner for j in reversed(range(t_start, t_end))]
        batch = s.latents.shape[0]
        s.chunk_times = s.timestep_schedule[s.chunk_step_indices][None].expand(batch, -1)
        s.next_chunk_times = s.timestep_schedule[[j + 1 for j in s.chunk_step_indices]][None].expand(batch, -1)
        s.refresh_cache = s.chunk_start > s.prefix_chunks and inner == 0
        first = s.chunk_start - int(s.refresh_cache)
        s.latent_model_input = s.latents[:, :, first * s.chunk_width : s.chunk_end * s.chunk_width]
        s.window_prompt_embeds = s.prompt_embeds[:, s.chunk_start : s.chunk_end]
        s.window_prompt_attention_mask = s.prompt_attention_mask[:, s.chunk_start : s.chunk_end]
        s.model_times = s.chunk_times
        limits = [
            s.noise2clean_kvrange[j // (s.num_inference_steps // len(s.noise2clean_kvrange))]
            for j in s.chunk_step_indices
        ]
        if s.refresh_cache:
            s.model_times = torch.cat([s.latents.new_full((batch, 1), s.clean_t), s.chunk_times], dim=1)
            s.window_prompt_embeds = torch.cat([s.negative_prompt_embeds[:, None], s.window_prompt_embeds], dim=1)
            s.window_prompt_attention_mask = torch.cat(
                [s.negative_prompt_attention_mask[:, None], s.window_prompt_attention_mask], dim=1
            )
            limits.insert(0, s.clean_chunk_kvrange)
        s.kv_ranges = _ranges(range(first, s.chunk_end), limits, s.chunk_tokens)
        return components, s


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
            _input(name, required=True)
            for name in [
                "latent_model_input",
                "model_times",
                "chunk_times",
                "window_prompt_embeds",
                "window_prompt_attention_mask",
                "negative_prompt_embeds",
                "negative_prompt_attention_mask",
                "kv_ranges",
                "refresh_cache",
                "chunk_start",
                "chunk_end",
                "chunk_width",
                "chunk_tokens",
                "num_inference_steps",
            ]
        ] + [
            _input("clean_kv_cache", default=None),
            _input("attention_kwargs", default=None),
            _input("cache_device", default=None),
        ]

    @property
    def intermediate_outputs(self):
        return [_output("velocity"), _output("clean_kv_cache")]

    @torch.no_grad()
    def __call__(self, components, s, i):
        batch, channels, _, height, width = s.latent_model_input.shape
        count = s.chunk_end - s.chunk_start
        skip = int(s.refresh_cache) * s.chunk_width
        independent = s.latent_model_input[:, :, skip:].reshape(batch, channels, count, s.chunk_width, height, width)
        independent = independent.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * count, channels, s.chunk_width, height, width
        )
        guider = components.guider
        guider.set_state(step=i, num_inference_steps=s.num_inference_steps, timestep=s.chunk_times)
        branches = guider.prepare_inputs(
            {
                "hidden_states": (s.latent_model_input, s.latent_model_input, independent),
                "encoder_hidden_states": (
                    s.window_prompt_embeds,
                    s.negative_prompt_embeds,
                    s.negative_prompt_embeds.repeat_interleave(count, dim=0),
                ),
                "encoder_attention_mask": (
                    s.window_prompt_attention_mask,
                    s.negative_prompt_attention_mask,
                    s.negative_prompt_attention_mask.repeat_interleave(count, dim=0),
                ),
                "timestep": (s.model_times, s.model_times, s.chunk_times.reshape(-1, 1)),
            }
        )
        refreshed_cache = None
        for branch_index, branch in enumerate(branches):
            guider.prepare_models(components.transformer)
            try:
                result = components.transformer(
                    hidden_states=branch.hidden_states,
                    encoder_hidden_states=branch.encoder_hidden_states,
                    encoder_attention_mask=branch.encoder_attention_mask,
                    timestep=branch.timestep,
                    caption_dropout_mask=torch.full(
                        (1,),
                        branch_index != 0,
                        device=independent.device,
                        dtype=torch.bool,
                    ),
                    kv_ranges=s.kv_ranges if branch_index < 2 else None,
                    kv_cache=s.clean_kv_cache if branch_index < 2 else None,
                    use_cache=branch_index == 1 and s.refresh_cache,
                    cache_token_count=s.chunk_start * s.chunk_tokens
                    if branch_index == 1 and s.refresh_cache
                    else None,
                    cache_device=s.cache_device if branch_index == 1 and s.refresh_cache else None,
                    attention_kwargs=s.attention_kwargs,
                )
            finally:
                guider.cleanup_models(components.transformer)
            if branch_index < 2:
                branch.noise_pred = result.sample[:, :, skip:]
            else:
                branch.noise_pred = (
                    result.sample.reshape(batch, count, channels, s.chunk_width, height, width)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(batch, channels, count * s.chunk_width, height, width)
                )
            if branch_index == 1 and s.refresh_cache:
                refreshed_cache = result.kv_cache
            del result
        s.velocity = guider(branches).pred
        if refreshed_cache is not None:
            s.clean_kv_cache = refreshed_cache
        return components, s


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
            _input(name, required=True)
            for name in [
                "latents",
                "velocity",
                "chunk_times",
                "next_chunk_times",
                "chunk_start",
                "chunk_end",
                "chunk_width",
                "chunk_step_indices",
                "num_inference_steps",
                "completed_chunks",
            ]
        ]

    @property
    def intermediate_outputs(self):
        return [_output("latents"), _output("completed_chunks")]

    def __call__(self, components, s, i):
        start, end = s.chunk_start * s.chunk_width, s.chunk_end * s.chunk_width
        s.latents[:, :, start:end] = components.scheduler.step(
            s.velocity, s.chunk_times, s.latents[:, :, start:end], next_timestep=s.next_chunk_times
        ).prev_sample
        if s.chunk_step_indices[0] == s.num_inference_steps - 1:
            s.completed_chunks.append(s.chunk_start)
        return components, s


# auto_docstring
class MagiDenoiseLoop(LoopSequentialPipelineBlocks):
    """
    Run the asynchronous MAGI chunk-denoising window with a clean-prefix cache.

      Components:
          transformer (`MagiTransformer3DModel`) guider (`MagiClassifierFreeGuidance`) scheduler (`MagiEulerScheduler`)

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
              Positive attention-window length used when recomputing clean chunks.
          clean_t (`float`):
              Model evaluation time for clean-prefix cache extraction.
          timestep_schedule (`Tensor`):
              FP32 schedule including the final Euler integration endpoint.
          clean_kv_cache (`tuple`, *optional*):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final
              generated chunk.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.

      Outputs:
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.
    """

    model_name = "magi"
    block_classes = [MagiLoopBeforeDenoiser, MagiLoopDenoiser, MagiLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self):
        return "Run the asynchronous MAGI chunk-denoising window with a clean-prefix cache."

    @property
    def loop_inputs(self):
        return [_input("num_window_steps", required=True)]

    @torch.no_grad()
    def __call__(self, components, state):
        s = self.get_block_state(state)
        with self.progress_bar(total=s.num_window_steps) as progress:
            for i in range(s.num_window_steps):
                components, s = self.loop_step(components, s, i=i)
                progress.update()
        self.set_block_state(state, s)
        return components, state


# auto_docstring
class MagiDenoiseStep(SequentialPipelineBlocks):
    """
    MAGI base-model latent generation; text encoding, partial-chunk prefixes, and VAE decoding are not included.

      Components:
          transformer (`MagiTransformer3DModel`) scheduler (`MagiEulerScheduler`) guider (`MagiClassifierFreeGuidance`)

      Inputs:
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
          prefix_latents (`Tensor`, *optional*):
              Optional full-chunk clean prefix; replaces the leading latent slots and remains unchanged.
          num_inference_steps (`int`, *optional*, defaults to 64):
              Number of Euler updates per generated chunk.
          chunk_width (`int`, *optional*, defaults to 6):
              Number of latent frames in each chunk.
          window_size (`int`, *optional*, defaults to 4):
              Maximum number of simultaneously denoised chunks.
          noise2clean_kvrange (`tuple`, *optional*, defaults to (5, 4, 3, 2)):
              Positive attention-window lengths in chunks, from early to late denoising stages.
          clean_chunk_kvrange (`int`, *optional*, defaults to 1):
              Positive attention-window length used when recomputing clean chunks.
          clean_t (`float`, *optional*, defaults to 0.9999):
              Model evaluation time for clean-prefix cache extraction.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.

      Outputs:
          num_chunks (`int`):
              Total number of latent chunks, including the supplied prefix.
          prefix_chunks (`int`):
              Number of supplied full-chunk prefix chunks.
          chunk_tokens (`int`):
              Number of Transformer tokens per latent chunk.
          steps_per_stage (`int`):
              Number of iterations before the active chunk window moves forward.
          num_window_steps (`int`):
              Total number of asynchronous window iterations.
          timestep_schedule (`Tensor`):
              FP32 schedule including the final Euler integration endpoint.
          clean_kv_cache (`tuple`):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final
              generated chunk.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.
          chunk_start (`int`):
              First active chunk index.
          chunk_end (`int`):
              Exclusive end index of the active chunk window.
          refresh_cache (`bool`):
              Whether this iteration prepends a finalized chunk to refresh its clean KV.
          chunk_step_indices (`list`):
              Denoising step indices for active chunks, ordered from oldest to newest.
          chunk_times (`Tensor`):
              Current per-chunk model times shaped (batch, active_chunks).
          next_chunk_times (`Tensor`):
              Next Euler endpoints shaped (batch, active_chunks).
          model_times (`Tensor`):
              Model times including an optional clean-refresh chunk.
          latent_model_input (`Tensor`):
              Current latent window including an optional clean-refresh chunk.
          window_prompt_embeds (`Tensor`):
              Conditional features for the current window, with null features for a clean-refresh chunk.
          window_prompt_attention_mask (`Tensor`):
              Boolean keep-mask matching the current window text features.
          kv_ranges (`tuple`):
              Exclusive token attention ranges indexing the cached prefix plus the current window.
          velocity (`Tensor`):
              Three-way guided FP32 velocities for active chunks only.
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
    """

    model_name = "magi"
    block_classes = [MagiPrepareDenoiseStep, MagiDenoiseLoop]
    block_names = ["prepare", "denoise"]

    @property
    def description(self):
        return "MAGI base-model latent generation; text encoding, partial-chunk prefixes, and VAE decoding are not included."


class MagiPreparePrefixStep(ModularPipelineBlocks):
    model_name = "magi"

    @property
    def description(self):
        return "Select complete prefix chunks for clean-cache initialization."

    @property
    def inputs(self):
        return [
            _input("latents", required=True),
            _input("chunk_width", default=6),
            InputParam(
                "conditioning_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Scaled prefix matching the generated latent batch and device.",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [_output("prefix_latents")]

    def __call__(self, components, state):
        s = self.get_block_state(state)
        prefix = s.conditioning_latents
        if not isinstance(s.chunk_width, int) or isinstance(s.chunk_width, bool) or s.chunk_width <= 0:
            raise ValueError("chunk_width must be a positive integer.")
        if (
            not isinstance(prefix, torch.Tensor)
            or prefix.ndim != 5
            or s.latents.ndim != 5
            or prefix.shape[:2] != s.latents.shape[:2]
            or prefix.shape[3:] != s.latents.shape[3:]
            or not 0 < prefix.shape[2] < s.latents.shape[2]
            or prefix.device != s.latents.device
            or not prefix.is_floating_point()
            or not prefix.isfinite().all()
        ):
            raise ValueError(
                "conditioning_latents must be a finite prefix matching the latent batch, shape, and device."
            )
        length = prefix.shape[2] // s.chunk_width * s.chunk_width
        s.prefix_latents = prefix[:, :, :length] if length else None
        self.set_block_state(state, s)
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
        return super().intermediate_outputs + [_output("latents")]

    def __call__(self, components, s, i):
        components, s = super().__call__(components, s, i)
        first = (s.chunk_start - int(s.refresh_cache)) * s.chunk_width
        end = min(s.conditioning_latents.shape[2], s.chunk_end * s.chunk_width)
        if first < end:
            s.latent_model_input = s.latent_model_input.clone()
            s.latent_model_input[:, :, : end - first] = s.conditioning_latents[:, :, first:end]
            start = s.chunk_start * s.chunk_width
            if start < end:
                # Euler integrates from the injected prefix, but cache refresh must not overwrite finalized output.
                s.latents[:, :, start:end] = s.conditioning_latents[:, :, start:end]
        return components, s


# auto_docstring
class MagiPrefixDenoiseLoop(MagiDenoiseLoop):
    """
    Denoise with per-evaluation prefix injection and separate clean-prefix cache refresh.

      Components:
          transformer (`MagiTransformer3DModel`) guider (`MagiClassifierFreeGuidance`) scheduler (`MagiEulerScheduler`)

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
              Positive attention-window length used when recomputing clean chunks.
          clean_t (`float`):
              Model evaluation time for clean-prefix cache extraction.
          timestep_schedule (`Tensor`):
              FP32 schedule including the final Euler integration endpoint.
          conditioning_latents (`Tensor`):
              Original scaled prefix, reinjected before every model evaluation.
          clean_kv_cache (`tuple`, *optional*):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final
              generated chunk.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.

      Outputs:
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.
    """

    model_name = "magi"
    block_classes = [MagiPrefixLoopBeforeDenoiser, MagiLoopDenoiser, MagiLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self):
        return "Denoise with per-evaluation prefix injection and separate clean-prefix cache refresh."


# auto_docstring
class MagiPrefixDenoiseStep(SequentialPipelineBlocks):
    """
    Generate latent continuations from complete or partial-chunk prefixes.

      Components:
          transformer (`MagiTransformer3DModel`) scheduler (`MagiEulerScheduler`) guider (`MagiClassifierFreeGuidance`)

      Inputs:
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
          chunk_width (`int`, *optional*, defaults to 6):
              Number of latent frames in each chunk.
          conditioning_latents (`Tensor`):
              Scaled prefix matching the generated latent batch and device.
          prompt_embeds (`Tensor`):
              Prepared conditional text features, shared across chunks or provided per chunk.
          prompt_attention_mask (`Tensor`):
              Boolean keep-mask matching the conditional text features.
          negative_prompt_embeds (`Tensor`):
              Learned null-caption features shaped (batch, length, caption_channels), shared across chunks.
          negative_prompt_attention_mask (`Tensor`):
              Boolean keep-mask for the learned null-caption features.
          num_inference_steps (`int`, *optional*, defaults to 64):
              Number of Euler updates per generated chunk.
          window_size (`int`, *optional*, defaults to 4):
              Maximum number of simultaneously denoised chunks.
          noise2clean_kvrange (`tuple`, *optional*, defaults to (5, 4, 3, 2)):
              Positive attention-window lengths in chunks, from early to late denoising stages.
          clean_chunk_kvrange (`int`, *optional*, defaults to 1):
              Positive attention-window length used when recomputing clean chunks.
          clean_t (`float`, *optional*, defaults to 0.9999):
              Model evaluation time for clean-prefix cache extraction.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.

      Outputs:
          prefix_latents (`Tensor`):
              Optional full-chunk clean prefix; replaces the leading latent slots and remains unchanged.
          num_chunks (`int`):
              Total number of latent chunks, including the supplied prefix.
          prefix_chunks (`int`):
              Number of supplied full-chunk prefix chunks.
          chunk_tokens (`int`):
              Number of Transformer tokens per latent chunk.
          steps_per_stage (`int`):
              Number of iterations before the active chunk window moves forward.
          num_window_steps (`int`):
              Total number of asynchronous window iterations.
          timestep_schedule (`Tensor`):
              FP32 schedule including the final Euler integration endpoint.
          clean_kv_cache (`tuple`):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final
              generated chunk.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.
          chunk_start (`int`):
              First active chunk index.
          chunk_end (`int`):
              Exclusive end index of the active chunk window.
          refresh_cache (`bool`):
              Whether this iteration prepends a finalized chunk to refresh its clean KV.
          chunk_step_indices (`list`):
              Denoising step indices for active chunks, ordered from oldest to newest.
          chunk_times (`Tensor`):
              Current per-chunk model times shaped (batch, active_chunks).
          next_chunk_times (`Tensor`):
              Next Euler endpoints shaped (batch, active_chunks).
          model_times (`Tensor`):
              Model times including an optional clean-refresh chunk.
          latent_model_input (`Tensor`):
              Current latent window including an optional clean-refresh chunk.
          window_prompt_embeds (`Tensor`):
              Conditional features for the current window, with null features for a clean-refresh chunk.
          window_prompt_attention_mask (`Tensor`):
              Boolean keep-mask matching the current window text features.
          kv_ranges (`tuple`):
              Exclusive token attention ranges indexing the cached prefix plus the current window.
          latents (`Tensor`):
              FP32 latent state shaped (batch, channels, frames, height, width), including prefix slots.
          velocity (`Tensor`):
              Three-way guided FP32 velocities for active chunks only.
    """

    model_name = "magi"
    block_classes = [MagiPreparePrefixStep, MagiPrepareDenoiseStep, MagiPrefixDenoiseLoop]
    block_names = ["prepare_prefix", "prepare", "denoise"]

    @property
    def description(self):
        return "Generate latent continuations from complete or partial-chunk prefixes."
