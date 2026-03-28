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
from ...guiders import ClassifierFreeGuidance
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
        block_state.sigma = torch.tensor([t.item()], dtype=torch.float32)

        return components, block_state


class LTX2LoopDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    def __init__(
        self,
        guider_input_fields: dict[str, Any] = None,
        guider_name: str = "guider",
        guider_config: FrozenDict = None,
    ):
        """Initialize a denoiser block for LTX2 that handles dual video+audio outputs.

        Args:
            guider_input_fields: Dictionary mapping transformer argument names to block_state field names.
                Values can be tuples (conditional, unconditional) or strings (same for both).
            guider_name: Name of the guider component to use (default: "guider").
            guider_config: Config for the guider component (default: guidance_scale=4.0).
        """
        self._guider_name = guider_name
        if guider_config is None:
            guider_config = FrozenDict({"guidance_scale": 4.0})
        self._guider_config = guider_config
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
        super().__init__()

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                self._guider_name,
                ClassifierFreeGuidance,
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
            InputParam("guidance_rescale", default=0.0, type_hint=float),
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
        guider = getattr(components, self._guider_name)
        guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        guider_state = guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)

        use_cross_timestep = getattr(components.transformer.config, "use_cross_timestep", False)
        sigma_val = components.scheduler.sigmas[i]

        # Pass raw sigma to wrapper if available (avoids timestep/1000 round-trip precision loss)
        if hasattr(components.transformer, "_raw_sigma"):
            components.transformer._raw_sigma = sigma_val

        for guider_state_batch in guider_state:
            guider.prepare_models(components.transformer)
            cond_kwargs = guider_state_batch.as_dict()
            cond_kwargs = {
                k: v.to(block_state.dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond_kwargs.items()
                if k in self._guider_input_fields.keys()
            }
            # Drop all-ones attention masks — they're functionally no-op but trigger
            # a different SDPA kernel path (masked vs unmasked) with different bf16 rounding.
            # Reference passes context_mask=None for unmasked attention.
            for mask_key in ["encoder_attention_mask", "audio_encoder_attention_mask"]:
                mask = cond_kwargs.get(mask_key)
                if mask is not None and mask.ndim <= 2 and (mask == 1).all():
                    cond_kwargs[mask_key] = None

            video_timestep = block_state.video_timestep
            audio_timestep = block_state.audio_timestep

            with components.transformer.cache_context("cond_uncond"):
                noise_pred_video, noise_pred_audio = components.transformer(
                    hidden_states=block_state.latent_model_input.to(block_state.dtype),
                    audio_hidden_states=block_state.audio_latent_model_input.to(block_state.dtype),
                    timestep=video_timestep,
                    audio_timestep=audio_timestep,
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
                    **cond_kwargs,
                )

            # Convert to x0 for guidance.
            prediction_type = getattr(components.transformer, "prediction_type", "velocity")
            if prediction_type == "x0":
                # Model already outputs x0 — no conversion needed
                x0_video = noise_pred_video
                x0_audio = noise_pred_audio
            else:
                # Model outputs velocity — convert to x0 matching reference's to_denoised:
                # (sample.f32 - velocity.f32 * sigma_f32).to(sample.dtype)
                # Reference uses f32 sigma (from denoise_mask * sigma, both f32).
                x0_video = self._convert_velocity_to_x0(
                    block_state.latents.float(), noise_pred_video.float(), sigma_val
                ).to(block_state.latents.dtype)
                x0_audio = self._convert_velocity_to_x0(
                    block_state.audio_latents.float(), noise_pred_audio.float(), sigma_val
                ).to(block_state.audio_latents.dtype)

            guider_state_batch.noise_pred = x0_video
            guider_state_batch.noise_pred_audio = x0_audio

            # Sub-step checkpoint: save/load x0 per condition
            _ckpts = getattr(block_state, "_checkpoints", None)
            if _ckpts:
                from diffusers.modular_pipelines.ltx2._checkpoint_utils import _maybe_checkpoint
                cond_label = "cond" if guider_state_batch is guider_state[0] else "uncond"
                _maybe_checkpoint(_ckpts, f"step_{i}_{cond_label}_x0", {
                    "video": x0_video, "audio": x0_audio,
                })
                # Load support: inject reference x0 for this condition
                ckpt = _ckpts.get(f"step_{i}_{cond_label}_x0")
                if ckpt is not None and ckpt.load:
                    x0_video = ckpt.data["video"].to(x0_video)
                    x0_audio = ckpt.data["audio"].to(x0_audio)
                    guider_state_batch.noise_pred = x0_video
                    guider_state_batch.noise_pred_audio = x0_audio

            guider.cleanup_models(components.transformer)

        # Apply guidance in x0 space using reference formula:
        # cond + (scale - 1) * (cond - uncond)
        # This is mathematically equivalent to uncond + scale * (cond - uncond)
        # but produces different bf16 rounding.
        if len(guider_state) == 2:
            guidance_scale = guider.guidance_scale
            x0_video_cond = guider_state[0].noise_pred
            x0_video_uncond = guider_state[1].noise_pred
            guided_x0_video = x0_video_cond + (guidance_scale - 1) * (x0_video_cond - x0_video_uncond)

            x0_audio_cond = guider_state[0].noise_pred_audio
            x0_audio_uncond = guider_state[1].noise_pred_audio
            guided_x0_audio = x0_audio_cond + (guidance_scale - 1) * (x0_audio_cond - x0_audio_uncond)

            if block_state.guidance_rescale > 0:
                guided_x0_video = self._rescale_noise_cfg(
                    guided_x0_video,
                    guider_state[0].noise_pred,
                    block_state.guidance_rescale,
                )
                guided_x0_audio = self._rescale_noise_cfg(
                    guided_x0_audio,
                    x0_audio_cond,
                    block_state.guidance_rescale,
                )
        else:
            guided_x0_video = guider_state[0].noise_pred
            guided_x0_audio = guider_state[0].noise_pred_audio

        # Sub-step checkpoint: save/load guided x0
        _ckpts = getattr(block_state, "_checkpoints", None)
        if _ckpts:
            from diffusers.modular_pipelines.ltx2._checkpoint_utils import _maybe_checkpoint
            _maybe_checkpoint(_ckpts, f"step_{i}_guided_x0", {
                "video": guided_x0_video, "audio": guided_x0_audio,
            })
            # Load support: inject reference guided x0
            ckpt = _ckpts.get(f"step_{i}_guided_x0")
            if ckpt is not None and ckpt.load:
                guided_x0_video = ckpt.data["video"].to(guided_x0_video)
                guided_x0_audio = ckpt.data["audio"].to(guided_x0_audio)

        # Convert guided x0 back to velocity for the scheduler.
        # Use sigma_val.item() (Python float) to match reference's to_velocity which
        # does sigma.to(float32).item() — dividing by Python float vs 0-dim tensor
        # uses different CUDA kernels and can produce different results at specific values.
        sigma_scalar = sigma_val.item()
        block_state.noise_pred_video = self._convert_x0_to_velocity(
            block_state.latents.float(), guided_x0_video, sigma_scalar
        ).to(block_state.latents.dtype)
        block_state.noise_pred_audio = self._convert_x0_to_velocity(
            block_state.audio_latents.float(), guided_x0_audio, sigma_scalar
        ).to(block_state.audio_latents.dtype)

        return components, block_state

    @staticmethod
    def _rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg


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
