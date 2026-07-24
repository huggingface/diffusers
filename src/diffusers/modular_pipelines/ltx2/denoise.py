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

from ...models import LTX2VideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam


# Guidance / velocity-space helpers, mirrored from `diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline` and
# redefined here since modular blocks must not import from `diffusers.pipelines.*` (gotcha #1).
def convert_velocity_to_x0(
    sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler
) -> torch.Tensor:
    return sample - denoised_output * scheduler.sigmas[step_idx]


def convert_x0_to_velocity(
    sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler
) -> torch.Tensor:
    return (sample - denoised_output) / scheduler.sigmas[step_idx]


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    # Rescales `noise_cfg` toward the std of `noise_pred_text` to fix overexposure (https://hf.co/papers/2305.08891).
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


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
            "Joint video+audio denoiser with manual guidance. Runs the transformer once for the (batched) "
            "conditional/unconditional pass and optionally once each for spatio-temporal guidance (STG) and "
            "modality-isolation guidance, combining the per-modality x0 predictions via the delta formulation."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
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
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam("num_frames", type_hint=int, default=121),
            InputParam("frame_rate", type_hint=float, default=24.0),
            InputParam("guidance_scale", type_hint=float, default=4.0),
            InputParam("audio_guidance_scale", type_hint=float, default=None),
            InputParam("stg_scale", type_hint=float, default=0.0),
            InputParam("audio_stg_scale", type_hint=float, default=None),
            InputParam("modality_scale", type_hint=float, default=1.0),
            InputParam("audio_modality_scale", type_hint=float, default=None),
            InputParam("guidance_rescale", type_hint=float, default=0.0),
            InputParam("audio_guidance_rescale", type_hint=float, default=None),
            InputParam("spatio_temporal_guidance_blocks", type_hint=list, default=None),
            InputParam("use_cross_timestep", type_hint=bool, default=False),
            InputParam.template("attention_kwargs"),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        latents = block_state.latents
        audio_latents = block_state.audio_latents
        scheduler = components.scheduler
        audio_scheduler = block_state.audio_scheduler

        # Resolve audio guidance defaults (fall back to the video values).
        guidance_scale = block_state.guidance_scale
        audio_guidance_scale = block_state.audio_guidance_scale
        audio_guidance_scale = audio_guidance_scale if audio_guidance_scale is not None else guidance_scale
        stg_scale = block_state.stg_scale
        audio_stg_scale = block_state.audio_stg_scale if block_state.audio_stg_scale is not None else stg_scale
        modality_scale = block_state.modality_scale
        audio_modality_scale = (
            block_state.audio_modality_scale if block_state.audio_modality_scale is not None else modality_scale
        )
        guidance_rescale = block_state.guidance_rescale
        audio_guidance_rescale = (
            block_state.audio_guidance_rescale if block_state.audio_guidance_rescale is not None else guidance_rescale
        )
        do_cfg = guidance_scale > 1.0 or audio_guidance_scale > 1.0
        do_stg = stg_scale > 0.0 or audio_stg_scale > 0.0
        do_modality = modality_scale > 1.0 or audio_modality_scale > 1.0

        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        def _forward(
            hidden, audio_hidden, enc, audio_enc, mask, v_coords, a_coords, v_ts, a_ts, isolate, stg_blocks, ctx
        ):
            with components.transformer.cache_context(ctx):
                noise_pred_video, noise_pred_audio = components.transformer(
                    hidden_states=hidden,
                    audio_hidden_states=audio_hidden,
                    encoder_hidden_states=enc,
                    audio_encoder_hidden_states=audio_enc,
                    timestep=v_ts,
                    audio_timestep=a_ts,
                    sigma=a_ts,  # plain (unmasked) timestep, used by LTX-2.3
                    encoder_attention_mask=mask,
                    audio_encoder_attention_mask=mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=block_state.frame_rate,
                    audio_num_frames=block_state.audio_num_frames,
                    video_coords=v_coords,
                    audio_coords=a_coords,
                    isolate_modalities=isolate,
                    spatio_temporal_guidance_blocks=stg_blocks,
                    perturbation_mask=None,
                    use_cross_timestep=block_state.use_cross_timestep,
                    attention_kwargs=block_state.attention_kwargs,
                    return_dict=False,
                )
            return noise_pred_video.float(), noise_pred_audio.float()

        # Positive (conditional) single-batch conditioning, reused by the STG / modality passes.
        pos_enc = block_state.connector_prompt_embeds
        pos_audio_enc = block_state.connector_audio_prompt_embeds
        pos_mask = block_state.connector_attention_mask
        video_coords = block_state.video_coords
        audio_coords = block_state.audio_coords

        # 1. Main (conditional / unconditional) pass.
        if do_cfg:
            noise_pred_video, noise_pred_audio = _forward(
                torch.cat([block_state.latent_model_input] * 2),
                torch.cat([block_state.audio_latent_model_input] * 2),
                torch.cat([block_state.negative_connector_prompt_embeds, pos_enc]),
                torch.cat([block_state.negative_connector_audio_prompt_embeds, pos_audio_enc]),
                torch.cat([block_state.negative_connector_attention_mask, pos_mask]),
                video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1)),
                audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1)),
                torch.cat([block_state.video_timestep] * 2),
                torch.cat([block_state.audio_timestep] * 2),
                False,
                None,
                "cond_uncond",
            )
            noise_pred_video_uncond, noise_pred_video_cond = noise_pred_video.chunk(2)
            noise_pred_audio_uncond, noise_pred_audio_cond = noise_pred_audio.chunk(2)
            video_x0 = convert_velocity_to_x0(latents, noise_pred_video_cond, i, scheduler)
            video_uncond_x0 = convert_velocity_to_x0(latents, noise_pred_video_uncond, i, scheduler)
            audio_x0 = convert_velocity_to_x0(audio_latents, noise_pred_audio_cond, i, audio_scheduler)
            audio_uncond_x0 = convert_velocity_to_x0(audio_latents, noise_pred_audio_uncond, i, audio_scheduler)
            video_cfg_delta = (guidance_scale - 1) * (video_x0 - video_uncond_x0)
            audio_cfg_delta = (audio_guidance_scale - 1) * (audio_x0 - audio_uncond_x0)
        else:
            noise_pred_video, noise_pred_audio = _forward(
                block_state.latent_model_input,
                block_state.audio_latent_model_input,
                pos_enc,
                pos_audio_enc,
                pos_mask,
                video_coords,
                audio_coords,
                block_state.video_timestep,
                block_state.audio_timestep,
                False,
                None,
                "cond_uncond",
            )
            video_x0 = convert_velocity_to_x0(latents, noise_pred_video, i, scheduler)
            audio_x0 = convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)
            video_cfg_delta = audio_cfg_delta = 0

        # 2. Spatio-temporal guidance (extra pass with STG blocks perturbed).
        if do_stg:
            stg_video, stg_audio = _forward(
                block_state.latent_model_input,
                block_state.audio_latent_model_input,
                pos_enc,
                pos_audio_enc,
                pos_mask,
                video_coords,
                audio_coords,
                block_state.video_timestep,
                block_state.audio_timestep,
                False,
                block_state.spatio_temporal_guidance_blocks,
                "uncond_stg",
            )
            stg_video_x0 = convert_velocity_to_x0(latents, stg_video, i, scheduler)
            stg_audio_x0 = convert_velocity_to_x0(audio_latents, stg_audio, i, audio_scheduler)
            video_stg_delta = stg_scale * (video_x0 - stg_video_x0)
            audio_stg_delta = audio_stg_scale * (audio_x0 - stg_audio_x0)
        else:
            video_stg_delta = audio_stg_delta = 0

        # 3. Modality-isolation guidance (extra pass with A2V/V2A cross-attention disabled).
        if do_modality:
            mod_video, mod_audio = _forward(
                block_state.latent_model_input,
                block_state.audio_latent_model_input,
                pos_enc,
                pos_audio_enc,
                pos_mask,
                video_coords,
                audio_coords,
                block_state.video_timestep,
                block_state.audio_timestep,
                True,
                None,
                "uncond_modality",
            )
            mod_video_x0 = convert_velocity_to_x0(latents, mod_video, i, scheduler)
            mod_audio_x0 = convert_velocity_to_x0(audio_latents, mod_audio, i, audio_scheduler)
            video_modality_delta = (modality_scale - 1) * (video_x0 - mod_video_x0)
            audio_modality_delta = (audio_modality_scale - 1) * (audio_x0 - mod_audio_x0)
        else:
            video_modality_delta = audio_modality_delta = 0

        # 4. Combine guidance terms (in x0 space) and optionally rescale.
        video_g = video_x0 + video_cfg_delta + video_stg_delta + video_modality_delta
        audio_g = audio_x0 + audio_cfg_delta + audio_stg_delta + audio_modality_delta
        block_state.noise_pred_video = (
            rescale_noise_cfg(video_g, video_x0, guidance_rescale) if guidance_rescale > 0 else video_g
        )
        block_state.noise_pred_audio = (
            rescale_noise_cfg(audio_g, audio_x0, audio_guidance_rescale) if audio_guidance_rescale > 0 else audio_g
        )
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
