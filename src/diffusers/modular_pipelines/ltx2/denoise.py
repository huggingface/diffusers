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

from ...configuration_utils import FrozenDict
from ...models import LTX2VideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam
from .guider import LTX2Guidance, plan_guidance_passes


# Velocity-space helpers, mirrored from `diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline` and redefined here
# since modular blocks must not import from `diffusers.pipelines.*` (gotcha #1). The guidance combine itself
# (delta formulation + rescale) now lives in `LTX2Guidance`.
def convert_velocity_to_x0(
    sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler
) -> torch.Tensor:
    return sample - denoised_output * scheduler.sigmas[step_idx]


def convert_x0_to_velocity(
    sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler
) -> torch.Tensor:
    return (sample - denoised_output) / scheduler.sigmas[step_idx]


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = latents.shape
    latents = latents.reshape(
        batch_size,
        -1,
        num_frames // patch_size_t,
        patch_size_t,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


class LTX2LoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Text-to-video loop step that casts the video/audio latent inputs and expands the per-step timestep."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("audio_latents", type_hint=torch.Tensor, required=True),
            InputParam("dtype", type_hint=torch.dtype, required=True),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)
        block_state.audio_latent_model_input = block_state.audio_latents.to(block_state.dtype)
        timestep = t.expand(block_state.latents.shape[0])
        block_state.video_timestep = timestep
        block_state.audio_timestep = timestep
        return components, block_state


class LTX2Image2VideoLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Image-to-video loop step. Like the t2v variant, but zeroes the video timestep on the image-conditioned "
            "latent frame via the conditioning mask (audio timestep is unmasked)."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("audio_latents", type_hint=torch.Tensor, required=True),
            InputParam("conditioning_mask", type_hint=torch.Tensor, required=True),
            InputParam("dtype", type_hint=torch.dtype, required=True),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)
        block_state.audio_latent_model_input = block_state.audio_latents.to(block_state.dtype)
        timestep = t.expand(block_state.latents.shape[0])
        block_state.video_timestep = timestep.unsqueeze(-1) * (1 - block_state.conditioning_mask)
        block_state.audio_timestep = timestep
        return components, block_state


class LTX2LoopDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Joint video+audio denoiser. Runs the transformer once per guidance pass (each a single batch — no "
            "CFG concatenation), unioned across the video `guider` and audio `audio_guider`, converts each pass's "
            "velocity to x0, and delegates the per-modality guidance combine (CFG + STG + modality-isolation, in x0 "
            "space) to the two guiders."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec(
                "guider",
                LTX2Guidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
            ComponentSpec(
                "audio_guider",
                LTX2Guidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("audio_latents", type_hint=torch.Tensor, required=True),
            InputParam("audio_scheduler", required=True),
            InputParam("audio_num_frames", type_hint=int, required=True),
            InputParam("connector_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("connector_audio_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("connector_attention_mask", type_hint=torch.Tensor, required=True),
            InputParam("negative_connector_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("negative_connector_audio_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("negative_connector_attention_mask", type_hint=torch.Tensor, required=True),
            InputParam("video_coords", type_hint=torch.Tensor, required=True),
            InputParam("audio_coords", type_hint=torch.Tensor, required=True),
            InputParam.template("num_inference_steps", required=True),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=121),
            InputParam("frame_rate", type_hint=float, default=24.0),
            InputParam("use_cross_timestep", type_hint=bool, default=False),
            InputParam.template("attention_kwargs"),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        # Batch-invariant transformer kwargs, identical across every guidance pass.
        shared_kwargs = {
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": block_state.frame_rate,
            "audio_num_frames": block_state.audio_num_frames,
            "use_cross_timestep": block_state.use_cross_timestep,
            "attention_kwargs": block_state.attention_kwargs,
            "perturbation_mask": None,
        }

        def _predict(enc, audio_enc, mask, flags, ctx):
            with components.transformer.cache_context(ctx):
                noise_pred_video, noise_pred_audio = components.transformer(
                    hidden_states=block_state.latent_model_input,
                    audio_hidden_states=block_state.audio_latent_model_input,
                    encoder_hidden_states=enc,
                    audio_encoder_hidden_states=audio_enc,
                    encoder_attention_mask=mask,
                    audio_encoder_attention_mask=mask,
                    timestep=block_state.video_timestep,
                    audio_timestep=block_state.audio_timestep,
                    sigma=block_state.audio_timestep,  # plain (unmasked) timestep, used by LTX-2.3
                    video_coords=block_state.video_coords,
                    audio_coords=block_state.audio_coords,
                    isolate_modalities=flags["isolate_modalities"],
                    spatio_temporal_guidance_blocks=flags["spatio_temporal_guidance_blocks"],
                    return_dict=False,
                    **shared_kwargs,
                )
            return noise_pred_video.float(), noise_pred_audio.float()

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        components.audio_guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        # Run each pass once (single batch); convert velocity->x0 and stash per modality by pass identifier. The
        # union spans both guiders, so a pass runs if either modality needs it (the other just won't combine it).
        video_x0, audio_x0 = {}, {}
        for spec in plan_guidance_passes(components.guider, components.audio_guider):
            if spec["conditioning"] == "cond":
                enc, audio_enc, mask = (
                    block_state.connector_prompt_embeds,
                    block_state.connector_audio_prompt_embeds,
                    block_state.connector_attention_mask,
                )
            else:
                enc, audio_enc, mask = (
                    block_state.negative_connector_prompt_embeds,
                    block_state.negative_connector_audio_prompt_embeds,
                    block_state.negative_connector_attention_mask,
                )
            v_vel, a_vel = _predict(enc, audio_enc, mask, spec["flags"], spec["identifier"])
            video_x0[spec["identifier"]] = convert_velocity_to_x0(block_state.latents, v_vel, i, components.scheduler)
            audio_x0[spec["identifier"]] = convert_velocity_to_x0(
                block_state.audio_latents, a_vel, i, block_state.audio_scheduler
            )

        # Combine per modality via each guider (delta formulation + rescale, in x0 space). The denoiser leaves the
        # guided x0 in `noise_pred_video`/`noise_pred_audio`; the after block converts back to velocity and steps.
        identifier_key = LTX2Guidance._identifier_key
        video_state = components.guider.prepare_inputs_from_block_state(block_state, {})
        audio_state = components.audio_guider.prepare_inputs_from_block_state(block_state, {})
        for batch in video_state:
            batch.noise_pred = video_x0[getattr(batch, identifier_key)]
        for batch in audio_state:
            batch.noise_pred = audio_x0[getattr(batch, identifier_key)]

        block_state.noise_pred_video = components.guider(video_state)[0]
        block_state.noise_pred_audio = components.audio_guider(audio_state)[0]
        return components, block_state


class LTX2LoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Text-to-video loop step that converts the guided x0 predictions to velocity and steps both schedulers."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("audio_latents", type_hint=torch.Tensor, required=True),
            InputParam("audio_scheduler", required=True),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        noise_pred_video = convert_x0_to_velocity(
            block_state.latents, block_state.noise_pred_video, i, components.scheduler
        )
        noise_pred_audio = convert_x0_to_velocity(
            block_state.audio_latents, block_state.noise_pred_audio, i, block_state.audio_scheduler
        )
        block_state.latents = components.scheduler.step(noise_pred_video, t, block_state.latents, return_dict=False)[0]
        block_state.audio_latents = block_state.audio_scheduler.step(
            noise_pred_audio, t, block_state.audio_latents, return_dict=False
        )[0]
        return components, block_state


class LTX2Image2VideoLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Image-to-video loop step. Steps the video scheduler only on the non-conditioning frames (keeping the "
            "clean image frame) and steps the audio scheduler as usual."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("audio_latents", type_hint=torch.Tensor, required=True),
            InputParam("audio_scheduler", required=True),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=121),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        spatial_patch = components.transformer_spatial_patch_size
        temporal_patch = components.transformer_temporal_patch_size
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        # Video preds/latents are unpacked so the conditioning frame (index 0) can be excluded from the step.
        noise_pred_video = _unpack_latents(
            block_state.noise_pred_video, latent_num_frames, latent_height, latent_width, spatial_patch, temporal_patch
        )
        latents = _unpack_latents(
            block_state.latents, latent_num_frames, latent_height, latent_width, spatial_patch, temporal_patch
        )
        noise_pred_video = convert_x0_to_velocity(latents, noise_pred_video, i, components.scheduler)

        pred_latents = components.scheduler.step(noise_pred_video[:, :, 1:], t, latents[:, :, 1:], return_dict=False)[
            0
        ]
        latents = torch.cat([latents[:, :, :1], pred_latents], dim=2)
        block_state.latents = _pack_latents(latents, spatial_patch, temporal_patch)

        noise_pred_audio = convert_x0_to_velocity(
            block_state.audio_latents, block_state.noise_pred_audio, i, block_state.audio_scheduler
        )
        block_state.audio_latents = block_state.audio_scheduler.step(
            noise_pred_audio, t, block_state.audio_latents, return_dict=False
        )[0]
        return components, block_state


class LTX2DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iteratively denoises the video and audio latents over `timesteps`. The per-iteration "
            "steps are customized via the `sub_blocks` attribute."
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", type_hint=torch.Tensor, required=True),
            InputParam.template("num_inference_steps", required=True),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.num_warmup_steps = max(
            len(block_state.timesteps) - block_state.num_inference_steps * components.scheduler.order, 0
        )

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


class LTX2DenoiseStep(LTX2DenoiseLoopWrapper):
    block_classes = [LTX2LoopBeforeDenoiser, LTX2LoopDenoiser, LTX2LoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Text-to-video denoise step. Iterates `LTX2DenoiseLoopWrapper.__call__`, running per step:\n"
            " - `LTX2LoopBeforeDenoiser`\n - `LTX2LoopDenoiser`\n - `LTX2LoopAfterDenoiser`"
        )


class LTX2Image2VideoDenoiseStep(LTX2DenoiseLoopWrapper):
    block_classes = [LTX2Image2VideoLoopBeforeDenoiser, LTX2LoopDenoiser, LTX2Image2VideoLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Image-to-video denoise step. Iterates `LTX2DenoiseLoopWrapper.__call__`, running per step:\n"
            " - `LTX2Image2VideoLoopBeforeDenoiser`\n - `LTX2LoopDenoiser`\n - `LTX2Image2VideoLoopAfterDenoiser`"
        )
