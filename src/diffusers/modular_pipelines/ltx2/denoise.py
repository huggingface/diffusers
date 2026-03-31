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

from typing import Any

import torch

from ...configuration_utils import FrozenDict
from ...guiders import LTX2MultiModalGuidance
from ...models.transformers import LTX2VideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam


logger = logging.get_logger(__name__)


class LTX2LoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that prepares the latent inputs for the denoiser, "
            "including timestep masking for conditioned frames."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("audio_latents", required=True, type_hint=torch.Tensor),
            InputParam("dtype", required=True, type_hint=torch.dtype),
            InputParam("conditioning_mask", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)
        block_state.audio_latent_model_input = block_state.audio_latents.to(block_state.dtype)

        batch_size = block_state.latent_model_input.shape[0]
        num_video_tokens = block_state.latent_model_input.shape[1]
        num_audio_tokens = block_state.audio_latent_model_input.shape[1]

        video_timestep = t.expand(batch_size, num_video_tokens)

        if block_state.conditioning_mask is not None:
            block_state.video_timestep = video_timestep * (
                1 - block_state.conditioning_mask.squeeze(-1)
            )
        else:
            block_state.video_timestep = video_timestep

        block_state.audio_timestep = t.expand(batch_size, num_audio_tokens)
        # Sigma for prompt_adaln: f32 to match reference's f32(sigma * scale_multiplier)
        block_state.sigma = torch.tensor([t.item()], dtype=torch.float32, device=t.device)

        return components, block_state


class LTX2LoopDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    _default_guider_config = FrozenDict({
        "guidance_scale": 3.0,
        "audio_guidance_scale": 7.0,
        "skip_layer_guidance_scale": 1.0,
        "spatio_temporal_guidance_blocks": [28],
        "modality_guidance_scale": 3.0,
        "guidance_rescale": 0.7,
    })

    def __init__(
        self,
        guider_input_fields: dict[str, Any] = None,
        guider_config: FrozenDict = None,
    ):
        """Initialize a denoiser block for LTX2 that handles dual video+audio outputs.

        Uses [`LTX2MultiModalGuidance`] which handles CFG, STG (via hook-based attention
        skip), modality isolation, and rescaling — with separate scales for video and audio.

        Args:
            guider_input_fields: Dictionary mapping transformer argument names to block_state field names.
                Values can be tuples (conditional, unconditional) or strings (same for both).
            guider_config: Config for the LTX2MultiModalGuidance guider.
        """
        if guider_input_fields is None:
            guider_input_fields = {
                "encoder_hidden_states": ("connector_prompt_embeds", "connector_negative_prompt_embeds"),
                "audio_encoder_hidden_states": (
                    "connector_audio_prompt_embeds",
                    "connector_audio_negative_prompt_embeds",
                ),
                "encoder_attention_mask": ("connector_attention_mask", "connector_negative_attention_mask"),
                "audio_encoder_attention_mask": ("connector_attention_mask", "connector_negative_attention_mask"),
            }
        if not isinstance(guider_input_fields, dict):
            raise ValueError(f"guider_input_fields must be a dictionary but is {type(guider_input_fields)}")
        self._guider_input_fields = guider_input_fields
        self._guider_config = guider_config or self._default_guider_config
        super().__init__()

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "guider",
                LTX2MultiModalGuidance,
                config=self._guider_config,
                default_creation_method="from_config",
            ),
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that runs the transformer with guidance "
            "and handles dual video+audio output splitting. CFG is applied in x0 space "
            "to match the reference implementation."
        )

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam("attention_kwargs"),
            InputParam("num_inference_steps", required=True, type_hint=int),
            InputParam("latent_num_frames", required=True, type_hint=int),
            InputParam("latent_height", required=True, type_hint=int),
            InputParam("latent_width", required=True, type_hint=int),
            InputParam("audio_num_frames", required=True, type_hint=int),
            InputParam("frame_rate", default=24.0, type_hint=float),
            InputParam("video_coords", required=True, type_hint=torch.Tensor),
            InputParam("audio_coords", required=True, type_hint=torch.Tensor),
            InputParam("sigma", type_hint=torch.Tensor),
        ]
        guider_input_names = []
        for value in self._guider_input_fields.values():
            if isinstance(value, tuple):
                guider_input_names.extend(value)
            else:
                guider_input_names.append(value)

        for name in set(guider_input_names):
            inputs.append(InputParam(name=name, type_hint=torch.Tensor))
        return inputs

    @staticmethod
    def _convert_velocity_to_x0(sample, velocity, sigma):
        return sample - velocity * sigma

    @staticmethod
    def _convert_x0_to_velocity(sample, x0, sigma):
        return (sample - x0) / sigma

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        guider = components.guider
        guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        guider_state = guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        sigma_val = components.scheduler.sigmas[i]
        use_cross_timestep = getattr(components.transformer.config, "use_cross_timestep", False)

        transformer_kwargs = dict(
            hidden_states=block_state.latent_model_input.to(block_state.dtype),
            audio_hidden_states=block_state.audio_latent_model_input.to(block_state.dtype),
            timestep=block_state.video_timestep,
            audio_timestep=block_state.audio_timestep,
            sigma=block_state.sigma,
            num_frames=block_state.latent_num_frames,
            height=block_state.latent_height,
            width=block_state.latent_width,
            fps=block_state.frame_rate,
            audio_num_frames=block_state.audio_num_frames,
            video_coords=block_state.video_coords,
            audio_coords=block_state.audio_coords,
            use_cross_timestep=use_cross_timestep,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
        )

        # --- Passes 1-4: Cond + Uncond + STG + Modality (all via guider) ---
        # The guider handles hooks: STG (attention skip) and modality (cross-attn skip).
        for guider_state_batch in guider_state:
            guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {
                k: v.to(block_state.dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond_kwargs.items()
                if k in self._guider_input_fields.keys()
            }
            # Drop all-ones attention masks — they trigger a different SDPA kernel with different bf16 rounding.
            if cond_kwargs.get("encoder_attention_mask") is not None and cond_kwargs["encoder_attention_mask"].ndim <= 2 and (cond_kwargs["encoder_attention_mask"] == 1).all():
                cond_kwargs["encoder_attention_mask"] = None
            if cond_kwargs.get("audio_encoder_attention_mask") is not None and cond_kwargs["audio_encoder_attention_mask"].ndim <= 2 and (cond_kwargs["audio_encoder_attention_mask"] == 1).all():
                cond_kwargs["audio_encoder_attention_mask"] = None

            model_kwargs = getattr(guider_state_batch, "_model_kwargs", {})

            with components.transformer.cache_context("cond_uncond"):
                noise_pred_video, noise_pred_audio = components.transformer(**transformer_kwargs, **cond_kwargs, **model_kwargs)

            guider_state_batch.noise_pred = self._convert_velocity_to_x0(
                block_state.latents.float(), noise_pred_video.float(), sigma_val
            ).to(block_state.latents.dtype)
            guider_state_batch.noise_pred_audio = self._convert_velocity_to_x0(
                block_state.audio_latents.float(), noise_pred_audio.float(), sigma_val
            ).to(block_state.audio_latents.dtype)

            guider.cleanup_models(components.transformer)

        # --- Combine via guider (guider owns the formula) ---
        output = guider(guider_state)
        guided_x0_video = output.pred
        guided_x0_audio = output.pred_audio

        # Convert guided x0 back to velocity for the scheduler.
        sigma_scalar = sigma_val.item()
        block_state.noise_pred_video = self._convert_x0_to_velocity(
            block_state.latents.float(), guided_x0_video, sigma_scalar
        ).to(block_state.latents.dtype)
        block_state.noise_pred_audio = self._convert_x0_to_velocity(
            block_state.audio_latents.float(), guided_x0_audio, sigma_scalar
        ).to(block_state.audio_latents.dtype)

        return components, block_state


class LTX2LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop that updates latents via scheduler step, "
            "with optional x0-space conditioning blending."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("conditioning_mask", type_hint=torch.Tensor),
            InputParam("clean_latents", type_hint=torch.Tensor),
            InputParam("audio_scheduler", required=True),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        noise_pred_video = block_state.noise_pred_video
        noise_pred_audio = block_state.noise_pred_audio

        if block_state.conditioning_mask is not None:
            # x0 blending: convert velocity to x0, blend with clean latents, convert back
            sigma = components.scheduler.sigmas[i]
            denoised_sample = block_state.latents - noise_pred_video * sigma

            bsz = noise_pred_video.size(0)
            conditioning_mask = block_state.conditioning_mask[:bsz]
            clean_latents = block_state.clean_latents

            denoised_sample_cond = (
                denoised_sample * (1 - conditioning_mask) + clean_latents.float() * conditioning_mask
            ).to(noise_pred_video.dtype)

            denoised_latents_cond = ((block_state.latents - denoised_sample_cond) / sigma).to(
                noise_pred_video.dtype
            )
            block_state.latents = components.scheduler.step(
                denoised_latents_cond, t, block_state.latents, return_dict=False
            )[0]
        else:
            block_state.latents = components.scheduler.step(
                noise_pred_video, t, block_state.latents, return_dict=False
            )[0]

        block_state.audio_latents = block_state.audio_scheduler.step(
            noise_pred_audio, t, block_state.audio_latents, return_dict=False
        )[0]

        return components, block_state


class LTX2DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Pipeline block that iteratively denoises the latents over timesteps for LTX2"

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("num_inference_steps", required=True, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        _checkpoints = state.get("_checkpoints")

        block_state.num_warmup_steps = max(
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0
        )

        # Checkpoint: save/load preloop state
        if _checkpoints:
            from diffusers.modular_pipelines.ltx2._checkpoint_utils import _maybe_checkpoint
            _maybe_checkpoint(_checkpoints, "preloop", {
                "video_latent": block_state.latents, "audio_latent": block_state.audio_latents,
            })
            if "preloop" in _checkpoints and _checkpoints["preloop"].load:
                d = _checkpoints["preloop"].data
                block_state.latents = d["video_latent"].to(block_state.latents)
                block_state.audio_latents = d["audio_latent"].to(block_state.audio_latents)

        # Pass _checkpoints to sub-blocks via block_state
        if _checkpoints:
            block_state._checkpoints = _checkpoints

        try:
            with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
                for i, t in enumerate(block_state.timesteps):
                    components, block_state = self.loop_step(components, block_state, i=i, t=t)

                    # Checkpoint: save velocity (= guided prediction) after denoiser, before scheduler
                    if _checkpoints:
                        _maybe_checkpoint(_checkpoints, f"step_{i}_velocity", {
                            "video": block_state.noise_pred_video, "audio": block_state.noise_pred_audio,
                        })

                    # Checkpoint: save/load after each step
                    if _checkpoints:
                        _maybe_checkpoint(_checkpoints, f"after_step_{i}", {
                            "video_latent": block_state.latents, "audio_latent": block_state.audio_latents,
                        })
                        if f"after_step_{i}" in _checkpoints and _checkpoints[f"after_step_{i}"].load:
                            d = _checkpoints[f"after_step_{i}"].data
                            block_state.latents = d["video_latent"].to(block_state.latents)
                            block_state.audio_latents = d["audio_latent"].to(block_state.audio_latents)

                    if i == len(block_state.timesteps) - 1 or (
                        (i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0
                    ):
                        progress_bar.update()
        except StopIteration:
            pass

        self.set_block_state(state, block_state)
        return components, state


class LTX2DenoiseStep(LTX2DenoiseLoopWrapper):
    block_classes = [
        LTX2LoopBeforeDenoiser,
        LTX2LoopDenoiser(
            guider_input_fields={
                "encoder_hidden_states": ("connector_prompt_embeds", "connector_negative_prompt_embeds"),
                "audio_encoder_hidden_states": (
                    "connector_audio_prompt_embeds",
                    "connector_audio_negative_prompt_embeds",
                ),
                "encoder_attention_mask": ("connector_attention_mask", "connector_negative_attention_mask"),
                "audio_encoder_attention_mask": ("connector_attention_mask", "connector_negative_attention_mask"),
            }
        ),
        LTX2LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoises video and audio latents.\n"
            "At each iteration, it runs:\n"
            " - LTX2LoopBeforeDenoiser (prepare inputs, timestep masking)\n"
            " - LTX2LoopDenoiser (transformer forward + guidance)\n"
            " - LTX2LoopAfterDenoiser (scheduler step + x0 blending)\n"
            "Supports T2V, I2V, and conditional generation."
        )
