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

from ...models.transformers.transformer_hy_nomas import HyNomasTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HyNomasGenerationState, HyNomasModularPipeline


# auto_docstring
class HyNomasSegmentDenoiseStep(ModularPipelineBlocks):
    """
    Run flow-matching Euler steps for one segment, with earlier segments supplied as cached keys and values.

      Components:
          transformer (`HyNomasTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          initial_latents (`Tensor`):
              Initial noise for all new segments, shaped (1, channels, new_latent_frames, latent_height, latent_width).
          segment_latent_lengths (`list`):
              Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).
          segment_pixel_lengths (`list`):
              Exact visible pixel-frame count of each new segment after removing decoder padding from the final
              segment.
          prompt_embeds (`list`):
              Qwen hidden states for each segment, shaped (1, max_sequence_length, hidden_size), with the chat prefix
              removed.
          prompt_attention_masks (`list`):
              Per-segment text masks shaped (1, max_sequence_length), with one for valid tokens and zero for padding.
          segment_cameras (`list`):
              Normalized camera-to-world poses for each segment, shaped (1, latent_frames * 4, 4, 4), or None without
              camera conditioning.
          action_text_offsets (`list`):
              Per-segment action-token boundaries in the cropped text sequence, or None for prompts without structured
              actions.
          action_frame_windows (`list`):
              Per-segment (start, end, action_index) intervals in latent frames; -1 denotes an action gap, and None
              disables action-specific windows.
          cache_action_starts (`list`):
              Per-segment text boundary after which tokens are hidden from future segments; None keeps all valid text
              visible.
          session (`HyNomasGenerationState`):
              Forked world state supplying cached keys and values from earlier segments, including prior calls.
          num_inference_steps (`int`, *optional*, defaults to 8):
              Number of flow-matching Euler denoising steps for each new segment.
          cache_noise_sigma (`float`, *optional*, defaults to 0.206897):
              Noise level in [0, 1] applied to a completed segment before committing its transformer keys and values to
              history.
          generator (`Generator`, *optional*):
              Rollout generator used to draw the noise for each segment cache write.

      Outputs:
          segment_latents (`Tensor`):
              Denoised latents for the current segment, shaped (1, channels, segment_latent_frames, latent_height,
              latent_width).
          session (`HyNomasGenerationState`):
              World state updated with the completed segment, its visible pixel length, and its keys and values.
    """

    model_name = "hy-nomas"

    @property
    def description(self):
        return (
            "Run flow-matching Euler steps for one segment, with earlier segments supplied as cached keys and values."
        )

    @property
    def expected_components(self):
        return [
            ComponentSpec("transformer", HyNomasTransformer3DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self):
        return [
            InputParam(
                "initial_latents",
                type_hint=torch.Tensor,
                description="Initial noise for all new segments, shaped (1, channels, new_latent_frames, latent_height, latent_width).",
                required=True,
            ),
            InputParam(
                "segment_latent_lengths",
                type_hint=list[int],
                description="Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).",
                required=True,
            ),
            InputParam(
                "segment_pixel_lengths",
                type_hint=list[int],
                description="Exact visible pixel-frame count of each new segment after removing decoder padding from the final segment.",
                required=True,
            ),
            InputParam(
                "prompt_embeds",
                type_hint=list[torch.Tensor],
                description="Qwen hidden states for each segment, shaped (1, max_sequence_length, hidden_size), with the chat prefix removed.",
                required=True,
            ),
            InputParam(
                "prompt_attention_masks",
                type_hint=list[torch.Tensor],
                description="Per-segment text masks shaped (1, max_sequence_length), with one for valid tokens and zero for padding.",
                required=True,
            ),
            InputParam(
                "segment_cameras",
                type_hint=list[torch.Tensor | None],
                description="Normalized camera-to-world poses for each segment, shaped (1, latent_frames * 4, 4, 4), or None without camera conditioning.",
                required=True,
            ),
            InputParam(
                "action_text_offsets",
                type_hint=list[list[int] | None],
                description="Per-segment action-token boundaries in the cropped text sequence, or None for prompts without structured actions.",
                required=True,
            ),
            InputParam(
                "action_frame_windows",
                type_hint=list[list[tuple[int, int, int]] | None],
                description="Per-segment (start, end, action_index) intervals in latent frames; -1 denotes an action gap, and None disables action-specific windows.",
                required=True,
            ),
            InputParam(
                "cache_action_starts",
                type_hint=list[int | None],
                description="Per-segment text boundary after which tokens are hidden from future segments; None keeps all valid text visible.",
                required=True,
            ),
            InputParam(
                "session",
                type_hint=HyNomasGenerationState,
                description="Forked world state supplying cached keys and values from earlier segments, including prior calls.",
                required=True,
            ),
            InputParam(
                "num_inference_steps",
                type_hint=int,
                description="Number of flow-matching Euler denoising steps for each new segment.",
                default=8,
            ),
            InputParam(
                "cache_noise_sigma",
                type_hint=float,
                description="Noise level in [0, 1] applied to a completed segment before committing its transformer keys and values to history.",
                default=0.206897,
            ),
            InputParam(
                "generator",
                type_hint=torch.Generator,
                description="Rollout generator used to draw the noise for each segment cache write.",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "segment_latents",
                type_hint=torch.Tensor,
                description="Denoised latents for the current segment, shaped (1, channels, segment_latent_frames, latent_height, latent_width).",
            ),
            OutputParam(
                "session",
                type_hint=HyNomasGenerationState,
                description="World state updated with the completed segment, its visible pixel length, and its keys and values.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, s, segment_index) -> tuple[HyNomasModularPipeline, BlockState]:
        if s.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        if not 0 <= s.cache_noise_sigma <= 1:
            raise ValueError("cache_noise_sigma must lie in [0, 1].")
        start = sum(s.segment_latent_lengths[:segment_index])
        end = start + s.segment_latent_lengths[segment_index]
        latents = s.initial_latents[:, :, start:end].clone()
        # The released solver uses a grid ending at 1 / steps, followed by zero.
        sigmas = torch.linspace(1, 0, s.num_inference_steps + 1)[:-1].tolist()
        components.scheduler.set_timesteps(sigmas=sigmas, device=latents.device)
        mask = s.prompt_attention_masks[segment_index]
        if mask is not None and mask.all():
            mask = None
        model_inputs = {
            "encoder_hidden_states": s.prompt_embeds[segment_index].to(
                device=components._execution_device, dtype=components.transformer.dtype
            ),
            "encoder_attention_mask": None if mask is None else mask.to(components._execution_device),
            "kv_cache": s.session.kv_cache,
            "frame_offset": s.session.frame_offset,
            "camera_extrinsics": s.segment_cameras[segment_index],
            "action_text_offsets": (
                [s.action_text_offsets[segment_index]] if s.action_frame_windows[segment_index] is not None else None
            ),
            "action_frame_windows": s.action_frame_windows[segment_index],
            "cache_action_start": s.cache_action_starts[segment_index],
        }
        for timestep in components.scheduler.timesteps:
            velocity = components.transformer(
                hidden_states=latents, timestep=timestep.expand(1), **model_inputs, return_dict=False
            )[0]
            latents = components.scheduler.step(velocity.float(), timestep, latents.float(), return_dict=False)[0].to(
                latents.dtype
            )
        s.segment_latents = latents
        cache_latents = latents
        sigma = s.cache_noise_sigma
        if sigma:
            noise = randn_tensor(latents.shape, generator=s.generator, device=latents.device, dtype=latents.dtype)
            cache_latents = ((1 - sigma) * latents.float() + sigma * noise.float()).to(latents.dtype)
        components.transformer(
            hidden_states=cache_latents,
            timestep=torch.full((1,), sigma * 1000, device=latents.device, dtype=torch.float32),
            update_cache=True,
            **model_inputs,
            return_dict=False,
        )
        s.session.latent_chunks.append(latents)
        s.session.pixel_lengths.append(s.segment_pixel_lengths[segment_index])
        s.session.frame_offset += latents.shape[2]
        return components, s


# auto_docstring
class HyNomasSegmentLoop(LoopSequentialPipelineBlocks):
    """
    Generate successive segments and commit each completed segment to the caller's world history.

      Components:
          transformer (`HyNomasTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          initial_latents (`Tensor`):
              Initial noise for all new segments, shaped (1, channels, new_latent_frames, latent_height, latent_width).
          segment_latent_lengths (`list`):
              Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).
          segment_pixel_lengths (`list`):
              Exact visible pixel-frame count of each new segment after removing decoder padding from the final
              segment.
          prompt_embeds (`list`):
              Qwen hidden states for each segment, shaped (1, max_sequence_length, hidden_size), with the chat prefix
              removed.
          prompt_attention_masks (`list`):
              Per-segment text masks shaped (1, max_sequence_length), with one for valid tokens and zero for padding.
          segment_cameras (`list`):
              Normalized camera-to-world poses for each segment, shaped (1, latent_frames * 4, 4, 4), or None without
              camera conditioning.
          action_text_offsets (`list`):
              Per-segment action-token boundaries in the cropped text sequence, or None for prompts without structured
              actions.
          action_frame_windows (`list`):
              Per-segment (start, end, action_index) intervals in latent frames; -1 denotes an action gap, and None
              disables action-specific windows.
          cache_action_starts (`list`):
              Per-segment text boundary after which tokens are hidden from future segments; None keeps all valid text
              visible.
          session (`HyNomasGenerationState`):
              Forked world state supplying cached keys and values from earlier segments, including prior calls.
          num_inference_steps (`int`, *optional*, defaults to 8):
              Number of flow-matching Euler denoising steps for each new segment.
          cache_noise_sigma (`float`, *optional*, defaults to 0.206897):
              Noise level in [0, 1] applied to a completed segment before committing its transformer keys and values to
              history.
          generator (`Generator`, *optional*):
              Rollout generator used to draw the noise for each segment cache write.

      Outputs:
          segment_latents (`Tensor`):
              Denoised latents for the current segment, shaped (1, channels, segment_latent_frames, latent_height,
              latent_width).
          session (`HyNomasGenerationState`):
              World state updated with the completed segment, its visible pixel length, and its keys and values.
    """

    model_name = "hy-nomas"
    block_classes = [HyNomasSegmentDenoiseStep]
    block_names = ["denoise_segment"]

    @property
    def description(self):
        return "Generate successive segments and commit each completed segment to the caller's world history."

    @property
    def loop_intermediate_outputs(self):
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Concatenated denoised latents for the complete world history, including prior calls and every new segment.",
            ),
            OutputParam(
                "session",
                type_hint=HyNomasGenerationState,
                description="Updated world history and generator state, ready for continuation or branching in a later call.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state) -> tuple[HyNomasModularPipeline, PipelineState]:
        s = self.get_block_state(state)
        for index in range(len(s.segment_latent_lengths)):
            components, s = self.loop_step(components, s, segment_index=index)
        s.latents = torch.cat(s.session.latent_chunks, dim=2)
        s.session.generator_state = s.generator.get_state().clone()
        s.session.generator_device = str(s.generator.device)
        self.set_block_state(state, s)
        return components, state
