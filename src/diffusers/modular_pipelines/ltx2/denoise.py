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


import inspect

import torch

from ...configuration_utils import FrozenDict
from ...guiders import LTX2Guidance
from ...models import LTX2VideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam


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
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Packed noisy audio latents to denoise.",
            ),
            InputParam(
                "dtype", type_hint=torch.dtype, required=True, description="The dtype the model inputs are cast to."
            ),
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
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Packed noisy audio latents to denoise.",
            ),
            InputParam("conditioning_mask", type_hint=torch.Tensor, required=True),
            InputParam(
                "dtype", type_hint=torch.dtype, required=True, description="The dtype the model inputs are cast to."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)
        block_state.audio_latent_model_input = block_state.audio_latents.to(block_state.dtype)
        timestep = t.expand(block_state.latents.shape[0])
        block_state.video_timestep = timestep.unsqueeze(-1) * (1 - block_state.conditioning_mask)
        block_state.audio_timestep = timestep
        return components, block_state


class LTX2ConditionLoopBeforeDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Condition loop step. Like the text-to-video variant, but scales the per-token video timestep by "
            "`1 - conditioning_mask`, so first-frame and keyframe condition tokens are seen at a reduced noise "
            "level (zero for `strength=1`). The audio timestep is unmasked."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Packed noisy audio latents to denoise.",
            ),
            InputParam(
                "conditioning_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Packed per-token conditioning strengths of shape [B, S, 1].",
            ),
            InputParam(
                "dtype", type_hint=torch.dtype, required=True, description="The dtype the model inputs are cast to."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)
        block_state.audio_latent_model_input = block_state.audio_latents.to(block_state.dtype)
        timestep = t.expand(block_state.latents.shape[0])
        block_state.video_timestep = timestep.unsqueeze(-1) * (1 - block_state.conditioning_mask.squeeze(-1))
        block_state.audio_timestep = timestep
        return components, block_state


# Default per-pass conditioning map for `LTX2LoopDenoiser`: transformer argument -> block-state attribute names
# indexed [cond, uncond, stg, modality]. STG and modality-isolation reuse the conditional (positive) tensors and
# differ only in their per-pass model flags, which the denoiser sets after preparation.
_GUIDER_INPUT_FIELDS = {
    "encoder_hidden_states": (
        "connector_prompt_embeds",
        "negative_connector_prompt_embeds",
        "connector_prompt_embeds",
        "connector_prompt_embeds",
    ),
    "audio_encoder_hidden_states": (
        "connector_audio_prompt_embeds",
        "negative_connector_audio_prompt_embeds",
        "connector_audio_prompt_embeds",
        "connector_audio_prompt_embeds",
    ),
    "encoder_attention_mask": (
        "connector_attention_mask",
        "negative_connector_attention_mask",
        "connector_attention_mask",
        "connector_attention_mask",
    ),
    "audio_encoder_attention_mask": (
        "connector_attention_mask",
        "negative_connector_attention_mask",
        "connector_attention_mask",
        "connector_attention_mask",
    ),
}


class LTX2LoopDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Joint video+audio denoiser. Runs the transformer once per guidance pass (each a single batch), with "
            "each pass's conditioning assembled by the guiders via `prepare_inputs_from_block_state` (driven by "
            "`_GUIDER_INPUT_FIELDS`) and unioned across the video `guider` and audio `audio_guider`; the per-pass "
            "model flags (STG blocks, modality isolation) are set by identifier afterwards. Converts each pass's "
            "velocity to x0 and delegates the per-modality CFG + STG + modality-isolation combine to the two guiders."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", LTX2VideoTransformer3DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec(
                "guider",
                LTX2Guidance,
                config=FrozenDict(
                    {
                        "guidance_scale": 3.0,
                        "stg_scale": 1.0,
                        "modality_scale": 3.0,
                        "guidance_rescale": 0.7,
                        "spatio_temporal_guidance_blocks": [28],
                    }
                ),
                default_creation_method="from_config",
            ),
            ComponentSpec(
                "audio_guider",
                LTX2Guidance,
                config=FrozenDict(
                    {
                        "guidance_scale": 7.0,
                        "stg_scale": 1.0,
                        "modality_scale": 3.0,
                        "guidance_rescale": 0.7,
                    }
                ),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Packed noisy audio latents to denoise.",
            ),
            InputParam("audio_scheduler", required=True),
            # `audio_num_frames`, `video_coords`, `audio_coords` arrive tagged `denoiser_input_fields` upstream and
            # are collected from the tagged dict (filtered against the transformer signature) in `__call__`.
            InputParam.template("denoiser_input_fields"),
            InputParam.template("num_inference_steps", required=True),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                required=True,
                description="The number of frames in the generated video.",
            ),
            InputParam(
                "frame_rate", type_hint=float, default=24.0, description="Frames per second of the generated video."
            ),
            InputParam.template("attention_kwargs"),
        ]
        # The text conditioning the guiders read off block_state, per `_GUIDER_INPUT_FIELDS`. The negative tensors
        # exist only under classifier-free guidance.
        inputs += [
            InputParam(
                "connector_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Video-branch text conditioning (cond), expanded per prompt.",
            ),
            InputParam(
                "connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                required=True,
                description="Audio-branch text conditioning (cond), expanded per prompt.",
            ),
            InputParam(
                "connector_attention_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Binary text attention mask (cond), expanded per prompt.",
            ),
            InputParam(
                "negative_connector_prompt_embeds",
                type_hint=torch.Tensor,
                description="Video-branch text conditioning (uncond); read only under classifier-free guidance.",
            ),
            InputParam(
                "negative_connector_audio_prompt_embeds",
                type_hint=torch.Tensor,
                description="Audio-branch text conditioning (uncond); read only under classifier-free guidance.",
            ),
            InputParam(
                "negative_connector_attention_mask",
                type_hint=torch.Tensor,
                description="Binary text attention mask (uncond); read only under classifier-free guidance.",
            ),
        ]
        return inputs

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        # Batch-invariant transformer kwargs, identical across every guidance pass. The upstream-produced ones
        # (`audio_num_frames`, `video_coords`, `audio_coords`) arrive tagged `denoiser_input_fields`; collect them by
        # filtering the tagged dict against the transformer signature. The latent dims are computed here (not routed
        # through the tag) so their names don't clash with the pixel-space num_frames/height/width in state.
        transformer_args = set(inspect.signature(components.transformer.forward).parameters)
        shared_kwargs = {k: v for k, v in block_state.denoiser_input_fields.items() if k in transformer_args}
        shared_kwargs.update(
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            fps=block_state.frame_rate,
            use_cross_timestep=components.use_cross_timestep,
            attention_kwargs=block_state.attention_kwargs,
            perturbation_mask=None,
        )

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        components.audio_guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        # Each guider maps block-state conditioning into one identifier-tagged batch per active pass via
        # `_GUIDER_INPUT_FIELDS` (transformer arg -> per-pass block-state attribute names, indexed
        # [cond, uncond, stg, modality]). A pass runs if *either* modality wants it, so union both guiders' batches
        # by identifier (same identifier => identical conditioning, built from the same map).
        identifier_key = LTX2Guidance._identifier_key
        if any(
            "pred_uncond" in guider.active_predictions() for guider in (components.guider, components.audio_guider)
        ):
            missing = [
                name
                for name in (
                    "negative_connector_prompt_embeds",
                    "negative_connector_audio_prompt_embeds",
                    "negative_connector_attention_mask",
                )
                if getattr(block_state, name, None) is None
            ]
            if missing:
                raise ValueError(
                    f"The guider runs classifier-free guidance but the unconditional conditioning {missing} is "
                    "missing. The text encoder produces it when the pipeline's guider has classifier-free guidance "
                    "enabled; when running the blocks separately, pass a `negative_prompt` to the text encoder."
                )
        batches_by_id = {}
        for guider in (components.guider, components.audio_guider):
            for batch in guider.prepare_inputs_from_block_state(block_state, _GUIDER_INPUT_FIELDS):
                batches_by_id.setdefault(getattr(batch, identifier_key), batch)
        guider_state = list(batches_by_id.values())

        # Per-pass model flags are pass-identity constants, not block-state conditioning, so they ride here rather
        # than through the name-referenced field map. Keying off the identifier with a plain-conditional default
        # keeps this correct for any guider: one that emits only a subset of passes (e.g. `ClassifierFreeGuidance`
        # -> just `pred_cond`/`pred_uncond`) simply gets no STG blocks and no modality isolation.
        stg_blocks = components.guider.spatio_temporal_guidance_blocks
        pass_flags = {
            "pred_cond": (None, False),
            "pred_uncond": (None, False),
            "pred_cond_stg": (stg_blocks, False),
            "pred_cond_modality": (None, True),
        }
        for batch in guider_state:
            batch.spatio_temporal_guidance_blocks, batch.isolate_modalities = pass_flags.get(
                getattr(batch, identifier_key), (None, False)
            )

        # One single-batch forward per pass; store each modality's x0 prediction on the batch. `prepare_models` /
        # `cleanup_models` are the standard per-pass hook points -- no-ops here, since LTX-2 carries its
        # perturbations as transformer flags (set above) rather than hooks.
        #
        # Parity note. Running every pass (cond/uncond included) as its own single-batch forward -- rather than the
        # batched `torch.cat([latents] * 2)` the standard `LTX2Pipeline` uses -- means this does NOT match the
        # reference bitwise. GPU matmul is not batch-invariant, so `cond` computed alone differs from `cond` inside
        # a batch-of-2: ~1e-6/op in fp32, but ~1e-2/op in bf16, which the CFG delta and the sampler amplify to ~10%
        # mean-relative latent divergence. A batched cond+uncond forward would be fp32-bitwise but cannot drive the
        # guider API per-pass, so this trades bitwiseness for using the guider API end-to-end. Gate any parity check
        # against `LTX2Pipeline` on fp32 and treat bf16 as close-but-not-bitwise.
        for batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {name: getattr(batch, name) for name in _GUIDER_INPUT_FIELDS}
            cond_kwargs["spatio_temporal_guidance_blocks"] = batch.spatio_temporal_guidance_blocks
            cond_kwargs["isolate_modalities"] = batch.isolate_modalities
            with components.transformer.cache_context(getattr(batch, identifier_key)):
                noise_pred_video, noise_pred_audio = components.transformer(
                    hidden_states=block_state.latent_model_input,
                    audio_hidden_states=block_state.audio_latent_model_input,
                    timestep=block_state.video_timestep,
                    audio_timestep=block_state.audio_timestep,
                    sigma=block_state.audio_timestep,  # plain (unmasked) timestep, used by LTX-2.3
                    return_dict=False,
                    **cond_kwargs,
                    **shared_kwargs,
                )
            batch.video_pred = convert_velocity_to_x0(
                block_state.latents, noise_pred_video.float(), i, components.scheduler
            )
            batch.audio_pred = convert_velocity_to_x0(
                block_state.audio_latents, noise_pred_audio.float(), i, block_state.audio_scheduler
            )
            components.guider.cleanup_models(components.transformer)

        # Combine each modality via its own guider, filtered to that guider's active passes so the batch count
        # matches `num_conditions`. The guiders combine in x0 space; the after block converts back to velocity.
        def _combine(guider, field):
            batches = [b for b in guider_state if getattr(b, identifier_key) in guider.active_predictions()]
            for b in batches:
                b.noise_pred = getattr(b, field)
            return guider(batches)[0]

        block_state.noise_pred_video = _combine(components.guider, "video_pred")
        block_state.noise_pred_audio = _combine(components.audio_guider, "audio_pred")
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
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Packed noisy audio latents to denoise.",
            ),
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
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Packed noisy audio latents to denoise.",
            ),
            InputParam("audio_scheduler", required=True),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                required=True,
                description="The number of frames in the generated video.",
            ),
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


class LTX2ConditionLoopAfterDenoiser(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Condition loop step. Blends the guided x0 prediction with the clean condition latents through the "
            "`conditioning_mask` (in x0 space, matching the reference `post_process_latent`), converts back to "
            "velocity, and steps both schedulers. Unlike the image-to-video variant the video step covers the whole "
            "sequence: appended keyframe tokens stay in place and are pinned by the mask rather than by slicing."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Packed noisy audio latents to denoise.",
            ),
            InputParam("audio_scheduler", required=True),
            InputParam(
                "conditioning_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Packed per-token conditioning strengths of shape [B, S, 1].",
            ),
            InputParam(
                "clean_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Clean condition latents at conditioned positions, zeros elsewhere.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        # Conditioning strengths run from 0 (always use the denoised sample) to 1 (always use the condition), with
        # intermediate values specifying how strongly to follow the condition. Applied in x0 space, not velocity
        # space (which is what the transformer outputs).
        conditioning_mask = block_state.conditioning_mask
        denoised = (
            block_state.noise_pred_video * (1 - conditioning_mask) + block_state.clean_latents * conditioning_mask
        ).to(block_state.noise_pred_video.dtype)

        noise_pred_video = convert_x0_to_velocity(block_state.latents, denoised, i, components.scheduler)
        noise_pred_audio = convert_x0_to_velocity(
            block_state.audio_latents, block_state.noise_pred_audio, i, block_state.audio_scheduler
        )
        block_state.latents = components.scheduler.step(noise_pred_video, t, block_state.latents, return_dict=False)[0]
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


class LTX2ConditionDenoiseStep(LTX2DenoiseLoopWrapper):
    block_classes = [LTX2ConditionLoopBeforeDenoiser, LTX2LoopDenoiser, LTX2ConditionLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

    @property
    def description(self) -> str:
        return (
            "Condition denoise step. Iterates `LTX2DenoiseLoopWrapper.__call__`, running per step:\n"
            " - `LTX2ConditionLoopBeforeDenoiser`\n - `LTX2LoopDenoiser`\n - `LTX2ConditionLoopAfterDenoiser`"
        )
