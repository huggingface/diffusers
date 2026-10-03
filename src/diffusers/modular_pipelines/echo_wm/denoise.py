# Copyright 2026 The Echo-WM and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import inspect
import math

import torch

from ...configuration_utils import FrozenDict
from ...guiders import LTX2Guidance
from ...models import AutoencoderKLLTX2Audio, EchoWMTransformer3DModel
from ...models.transformers.transformer_echo_wm import EchoWMKVCache
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from ..ltx2.before_denoise import _pack_latents
from ..ltx2.denoise import (
    LTX2Image2VideoDenoiseStep,
    LTX2Image2VideoLoopBeforeDenoiser,
    LTX2LoopAfterDenoiser,
    LTX2LoopDenoiser,
)
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


ECHO_WM_FLASH_TIMESTEPS = (1000, 750, 500, 250)
ECHO_WM_FLASH_VIDEO_CHUNK_SIZE = 3
ECHO_WM_FLASH_AUDIO_PREFIX = 2
ECHO_WM_FLASH_AUDIO_CHUNK_SIZE = 25


def _clear_echo_wm_audio_caches(caches: EchoWMKVCache) -> None:
    """Undo audio-cache writes from the video-only sink warmup."""
    for layer_cache in caches:
        for name in ("audio_self", "audio_text"):
            cache = layer_cache[name]
            cache["key"] = None
            cache["value"] = None
            if "positions" in cache:
                cache["positions"] = None


class EchoWMLoopDenoiser(LTX2LoopDenoiser):
    """Match Echo-WM's unmasked text attention and x0 rounding."""

    model_name = "echo-wm"

    def __init__(self):
        # The reference does not propagate Gemma's padding mask into the Base transformer's text cross-attention.
        # Its connector outputs already carry the intended padded-token representation.
        super().__init__(
            guider_input_fields={
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
            }
        )

    # Adapted from diffusers.modular_pipelines.ltx2.denoise.LTX2LoopDenoiser.__call__ to pass separate video/audio
    # sigmas and convert velocity to x0 with Echo-WM's per-token denoising timesteps.
    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, i: int, t: torch.Tensor):
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        transformer_args = set(inspect.signature(components.transformer.forward).parameters)
        shared_kwargs = {k: v for k, v in block_state.denoiser_input_fields.items() if k in transformer_args}
        shared_kwargs.update(
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            fps=block_state.frame_rate,
            use_cross_timestep=block_state.use_cross_timestep,
            attention_kwargs=block_state.attention_kwargs,
            perturbation_mask=None,
        )

        components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        components.audio_guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)

        identifier_key = LTX2Guidance._identifier_key
        batches_by_id = {}
        for guider in (components.guider, components.audio_guider):
            for batch in guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields):
                batches_by_id.setdefault(getattr(batch, identifier_key), batch)
        guider_state = list(batches_by_id.values())

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

        for batch in guider_state:
            components.guider.prepare_models(components.transformer)
            cond_kwargs = {name: getattr(batch, name) for name in self._guider_input_fields}
            cond_kwargs["spatio_temporal_guidance_blocks"] = batch.spatio_temporal_guidance_blocks
            cond_kwargs["isolate_modalities"] = batch.isolate_modalities
            with components.transformer.cache_context(getattr(batch, identifier_key)):
                noise_pred_video, noise_pred_audio = components.transformer(
                    hidden_states=block_state.latent_model_input,
                    audio_hidden_states=block_state.audio_latent_model_input,
                    timestep=block_state.video_timestep,
                    audio_timestep=block_state.audio_timestep,
                    sigma=block_state.video_sigma,
                    audio_sigma=block_state.audio_sigma,
                    return_dict=False,
                    **cond_kwargs,
                    **shared_kwargs,
                )
            batch.video_pred = (
                block_state.latents.float() - noise_pred_video.float() * block_state.video_x0_timestep.float()
            ).to(block_state.latents.dtype)
            batch.audio_pred = (
                block_state.audio_latents.float() - noise_pred_audio.float() * block_state.audio_x0_timestep.float()
            ).to(block_state.audio_latents.dtype)
            components.guider.cleanup_models(components.transformer)

        def _combine(guider, field):
            batches = [
                batch for batch in guider_state if getattr(batch, identifier_key) in guider.active_predictions()
            ]
            for batch in batches:
                batch.noise_pred = getattr(batch, field)
            return guider(batches)[0]

        block_state.noise_pred_video = _combine(components.guider, "video_pred")
        block_state.noise_pred_audio = _combine(components.audio_guider, "audio_pred")
        return components, block_state


class EchoWMImage2VideoLoopBeforeDenoiser(LTX2Image2VideoLoopBeforeDenoiser):
    """Expand audio timesteps per token while retaining scalar cross-modal sigmas."""

    model_name = "echo-wm"

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        components, block_state = super().__call__(components, block_state, i, t)
        sigma = t.expand(block_state.latents.shape[0])
        block_state.video_sigma = sigma
        block_state.audio_sigma = sigma
        normalized_sigma = components.scheduler.sigmas[i]
        video_denoise_mask = 1 - block_state.conditioning_mask
        if video_denoise_mask.ndim == block_state.latents.ndim - 1:
            video_denoise_mask = video_denoise_mask.unsqueeze(-1)
        block_state.video_x0_timestep = video_denoise_mask * normalized_sigma
        block_state.audio_x0_timestep = (
            torch.ones((*block_state.audio_latents.shape[:2], 1), device=t.device, dtype=torch.float32)
            * normalized_sigma
        )
        block_state.audio_timestep = (
            torch.ones((*block_state.audio_latents.shape[:2], 1), device=t.device, dtype=t.dtype)
            * sigma[:, None, None]
        )
        return components, block_state


class EchoWMLoopAfterDenoiser(LTX2LoopAfterDenoiser):
    """Match the reference Euler step's intermediate velocity and sample rounding."""

    model_name = "echo-wm"

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        latents = block_state.latents
        audio_latents = block_state.audio_latents
        video_sigma = components.scheduler.sigmas[i].to(torch.float32).item()
        audio_sigma = block_state.audio_scheduler.sigmas[i].to(torch.float32).item()
        video_velocity = ((latents.float() - block_state.noise_pred_video.float()) / video_sigma).to(latents.dtype)
        audio_velocity = ((audio_latents.float() - block_state.noise_pred_audio.float()) / audio_sigma).to(
            audio_latents.dtype
        )
        block_state.latents = components.scheduler.step(video_velocity.float(), t, latents, return_dict=False)[0].to(
            latents.dtype
        )
        block_state.audio_latents = block_state.audio_scheduler.step(
            audio_velocity.float(), t, audio_latents, return_dict=False
        )[0].to(audio_latents.dtype)
        return components, block_state


class EchoWMBaseDenoiseStep(LTX2Image2VideoDenoiseStep):
    model_name = "echo-wm"
    block_classes = [EchoWMImage2VideoLoopBeforeDenoiser, EchoWMLoopDenoiser, EchoWMLoopAfterDenoiser]


def echo_wm_flash_sigmas(
    timesteps: tuple[int, ...] | list[int], scheduler: FlowMatchEulerDiscreteScheduler | None = None
) -> list[float]:
    """Resolve the distilled timestep IDs against Echo-WM's 1000-step LTX-2 schedule."""
    if not timesteps:
        raise ValueError("`timesteps` must contain at least one timestep.")

    scheduler = scheduler or FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=math.exp(2.05),
        shift_terminal=0.1,
    )
    if scheduler.config.num_train_timesteps != 1000:
        raise ValueError(
            "Echo-WM Flash expects a scheduler with `num_train_timesteps=1000`, got "
            f"{scheduler.config.num_train_timesteps}."
        )
    scheduler.set_timesteps(timesteps=list(range(1000, 0, -1)))
    indices = [1000 - int(timestep) for timestep in timesteps]
    if any(index < 0 or index >= scheduler.sigmas.numel() for index in indices):
        raise ValueError(f"Echo-WM Flash timesteps must be in [0, 1000], got {timesteps}.")
    sigmas = [float(scheduler.sigmas[index]) for index in indices]
    if any(current <= following for current, following in zip(sigmas, sigmas[1:])):
        raise ValueError(f"Echo-WM Flash sigmas must be strictly descending, got {sigmas}.")
    return sigmas


def _flash_layout(latent_video_frames: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    if latent_video_frames < 1 or (latent_video_frames - 1) % ECHO_WM_FLASH_VIDEO_CHUNK_SIZE:
        raise ValueError(
            "Echo-WM Flash requires a latent video length of `1 + 3 * n`; equivalently, `num_frames` must be "
            "`1 + 24 * n`."
        )
    video_blocks = [(0, 1)] + [
        (start, start + ECHO_WM_FLASH_VIDEO_CHUNK_SIZE)
        for start in range(1, latent_video_frames, ECHO_WM_FLASH_VIDEO_CHUNK_SIZE)
    ]
    audio_frames = ECHO_WM_FLASH_AUDIO_PREFIX + (len(video_blocks) - 1) * ECHO_WM_FLASH_AUDIO_CHUNK_SIZE
    audio_blocks = [(0, ECHO_WM_FLASH_AUDIO_PREFIX)] + [
        (start, start + ECHO_WM_FLASH_AUDIO_CHUNK_SIZE)
        for start in range(ECHO_WM_FLASH_AUDIO_PREFIX, audio_frames, ECHO_WM_FLASH_AUDIO_CHUNK_SIZE)
    ]
    return video_blocks, audio_blocks, audio_frames


class EchoWMFlashChunkDenoiser(ModularPipelineBlocks):
    """Denoise and commit one autoregressive Echo-WM Flash video/audio chunk."""

    model_name = "echo-wm-flash"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", EchoWMTransformer3DModel)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor),
            OutputParam("audio_num_frames", type_hint=int),
            OutputParam("batch_size", type_hint=int),
            OutputParam("dtype", type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, block_index: int):
        transformer = components.transformer
        video_block = block_state.video_blocks[block_index]
        audio_block = block_state.audio_blocks[block_index]
        video_start, video_end = video_block
        audio_start, audio_end = audio_block

        def model_forward(video_sample, video_sigma, audio_sample, audio_sigma, isolate_modalities=False):
            video_token_start = video_start * block_state.patches_per_frame
            video_token_end = video_end * block_state.patches_per_frame
            timestep = (
                torch.full(
                    (block_state.batch_size, video_token_end - video_token_start),
                    video_sigma,
                    device=block_state.device,
                    dtype=video_sample.dtype,
                )
                * transformer.config.timestep_scale_multiplier
            )
            audio_timestep = (
                torch.full(
                    (block_state.batch_size, audio_end - audio_start),
                    audio_sigma,
                    device=block_state.device,
                    dtype=audio_sample.dtype,
                )
                * transformer.config.timestep_scale_multiplier
            )
            velocity_video, velocity_audio = transformer(
                hidden_states=video_sample.to(transformer.dtype),
                audio_hidden_states=audio_sample.to(transformer.dtype),
                encoder_hidden_states=block_state.prompt.to(transformer.dtype),
                audio_encoder_hidden_states=block_state.audio_prompt.to(transformer.dtype),
                encoder_attention_mask=block_state.prompt_mask,
                audio_encoder_attention_mask=block_state.prompt_mask,
                timestep=timestep,
                audio_timestep=audio_timestep,
                # The reference wrapper fixes both global sigmas to one before its transformer preprocessor scales
                # them. Diffusers accepts already-scaled sigmas, so pass the transformer's scale multiplier here.
                sigma=torch.full(
                    (block_state.batch_size,),
                    transformer.config.timestep_scale_multiplier,
                    device=block_state.device,
                    dtype=video_sample.dtype,
                ),
                audio_sigma=torch.full(
                    (block_state.batch_size,),
                    transformer.config.timestep_scale_multiplier,
                    device=block_state.device,
                    dtype=audio_sample.dtype,
                ),
                video_coords=block_state.video_coords[:, :, video_token_start:video_token_end],
                audio_coords=block_state.audio_coords[:, :, audio_start:audio_end],
                ucpe_viewmats=block_state.ucpe_viewmats,
                ucpe_intrinsics=block_state.ucpe_intrinsics,
                kv_caches=block_state.caches,
                current_video_token_start=video_token_start,
                current_audio_token_start=audio_start,
                isolate_modalities=isolate_modalities,
                use_cross_timestep=True,
                return_dict=False,
            )
            # The reference forms per-token sigmas in the sample dtype before the FP32 x0 calculation.
            video_sigma = torch.full((), video_sigma, device=video_sample.device, dtype=video_sample.dtype).float()
            audio_sigma = torch.full((), audio_sigma, device=audio_sample.device, dtype=audio_sample.dtype).float()
            video_x0 = (video_sample.float() - velocity_video.float() * video_sigma).to(video_sample.dtype)
            audio_x0 = (audio_sample.float() - velocity_audio.float() * audio_sigma).to(audio_sample.dtype)
            return video_x0, audio_x0

        def advance(denoised, next_sigma):
            noise = randn_tensor(
                denoised.shape,
                generator=block_state.generator,
                device=block_state.device,
                dtype=denoised.dtype,
            )
            return (1 - next_sigma) * denoised + next_sigma * noise

        if block_index == 0:
            video_sample = block_state.clean_image
        else:
            video_sample = block_state.initial_video[
                :, video_start * block_state.patches_per_frame : video_end * block_state.patches_per_frame
            ]
        audio_sample = block_state.initial_audio[:, audio_start:audio_end]
        if block_index == 0:
            # The reference primes the clean-image sink with no audio modality. The Diffusers transformer currently
            # requires audio inputs, so isolate the modalities and discard the incidental audio self/text-cache writes.
            model_forward(video_sample, 0.0, audio_sample, block_state.sigmas[0], True)
            _clear_echo_wm_audio_caches(block_state.caches)

        for step, sigma in enumerate(block_state.sigmas):
            denoised_video, denoised_audio = model_forward(
                video_sample,
                sigma if block_index else 0.0,
                audio_sample,
                sigma,
            )
            if step == len(block_state.sigmas) - 1:
                video_sample, audio_sample = denoised_video, denoised_audio
            else:
                video_sample = (
                    block_state.clean_image
                    if block_index == 0
                    else advance(denoised_video, block_state.sigmas[step + 1])
                )
                audio_sample = advance(denoised_audio, block_state.sigmas[step + 1])

        model_forward(video_sample, 0.0, audio_sample, 0.0)
        block_state.video_output[
            :, video_start * block_state.patches_per_frame : video_end * block_state.patches_per_frame
        ] = video_sample
        block_state.audio_output[:, audio_start:audio_end] = audio_sample
        block_state.latents = block_state.video_output
        block_state.audio_latents = block_state.audio_output
        return components, block_state


class EchoWMFlashDenoiseStep(LoopSequentialPipelineBlocks):
    """Run Echo-WM Flash's four-step autoregressive AV rollout with bounded sink-plus-FIFO KV caches."""

    model_name = "echo-wm-flash"
    block_classes = [EchoWMFlashChunkDenoiser]
    block_names = ["chunk_denoiser"]

    @property
    def description(self) -> str:
        return "Autoregressively denoise and commit each Echo-WM Flash video/audio chunk."

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", EchoWMTransformer3DModel),
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
            ComponentSpec(
                "scheduler",
                FlowMatchEulerDiscreteScheduler,
                config=FrozenDict(
                    {
                        "num_train_timesteps": 1000,
                        "shift": math.exp(2.05),
                        "use_dynamic_shifting": False,
                        "shift_terminal": 0.1,
                    }
                ),
                default_creation_method="from_config",
            ),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("image_latents", type_hint=torch.Tensor, required=True),
            InputParam("connector_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("connector_audio_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("connector_attention_mask", type_hint=torch.Tensor, required=True),
            InputParam("ucpe_viewmats", type_hint=torch.Tensor, required=True),
            InputParam("ucpe_intrinsics", type_hint=torch.Tensor, required=True),
            InputParam.template("height", default=704),
            InputParam.template("width", default=1280),
            InputParam("num_frames", type_hint=int, default=241),
            InputParam("frame_rate", type_hint=float, default=24.0),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam.template("generator"),
            InputParam(
                "batch_size",
                type_hint=int,
                required=True,
                description="The number of prompts before per-prompt video expansion.",
            ),
            InputParam(
                "timesteps",
                type_hint=list,
                default=list(ECHO_WM_FLASH_TIMESTEPS),
                description="Distilled denoising timestep IDs for each autoregressive chunk.",
            ),
            InputParam(
                "video_cache_size",
                type_hint=int,
                default=19,
                description="Maximum number of latent video frames retained in the bounded KV cache.",
            ),
            InputParam(
                "video_sink_size",
                type_hint=int,
                default=7,
                description="Number of leading latent video frames permanently retained as the cache sink.",
            ),
        ]

    @property
    def loop_intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor),
            OutputParam("audio_num_frames", type_hint=int),
            OutputParam("batch_size", type_hint=int),
            OutputParam("dtype", type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        transformer = components.transformer
        if components.transformer_spatial_patch_size != 1 or components.transformer_temporal_patch_size != 1:
            raise ValueError("Echo-WM Flash currently requires transformer patch sizes of 1.")

        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        latent_video_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        if block_state.num_frames != (latent_video_frames - 1) * components.vae_temporal_compression_ratio + 1:
            raise ValueError("Echo-WM Flash requires `num_frames` to be `1 + 8 * n`.")
        video_blocks, audio_blocks, audio_num_frames = _flash_layout(latent_video_frames)
        patches_per_frame = latent_height * latent_width
        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        latent_dtype = transformer.dtype

        prompt = block_state.connector_prompt_embeds.repeat_interleave(block_state.num_videos_per_prompt, dim=0)
        audio_prompt = block_state.connector_audio_prompt_embeds.repeat_interleave(
            block_state.num_videos_per_prompt, dim=0
        )
        prompt_mask = block_state.connector_attention_mask.repeat_interleave(block_state.num_videos_per_prompt, dim=0)
        image_latents = block_state.image_latents
        if batch_size % image_latents.shape[0]:
            raise ValueError(
                f"The image batch size ({image_latents.shape[0]}) must divide the requested batch size ({batch_size})."
            )
        image_latents = image_latents.repeat_interleave(batch_size // image_latents.shape[0], dim=0)
        clean_image = _pack_latents(image_latents)[:, :patches_per_frame].to(device=device, dtype=latent_dtype)

        video_shape = (batch_size, latent_video_frames * patches_per_frame, transformer.config.in_channels)
        initial_video = randn_tensor(video_shape, generator=block_state.generator, device=device, dtype=latent_dtype)
        audio_channels = components.audio_vae.config.latent_channels
        audio_mel_bins = components.audio_vae.config.mel_bins // components.audio_vae_mel_compression_ratio
        initial_audio = randn_tensor(
            (batch_size, audio_num_frames, audio_channels * audio_mel_bins),
            generator=block_state.generator,
            device=device,
            dtype=latent_dtype,
        )
        video_output = torch.zeros_like(initial_video)
        video_output[:, :patches_per_frame] = clean_image
        audio_output = torch.zeros_like(initial_audio)

        video_coords = transformer.rope.prepare_video_coords(
            batch_size, latent_video_frames, latent_height, latent_width, device, fps=block_state.frame_rate
        ).to(latent_dtype)
        audio_coords = transformer.audio_rope.prepare_audio_coords(batch_size, audio_num_frames, device)

        if not 0 < block_state.video_sink_size < block_state.video_cache_size:
            raise ValueError("Expected `0 < video_sink_size < video_cache_size`.")
        if (block_state.video_cache_size - 1) % ECHO_WM_FLASH_VIDEO_CHUNK_SIZE or (
            block_state.video_sink_size - 1
        ) % ECHO_WM_FLASH_VIDEO_CHUNK_SIZE:
            raise ValueError("Flash video cache and sink sizes must be `1 + 3 * n`.")
        audio_cache_size = (
            ECHO_WM_FLASH_AUDIO_PREFIX
            + ((block_state.video_cache_size - 1) // ECHO_WM_FLASH_VIDEO_CHUNK_SIZE) * ECHO_WM_FLASH_AUDIO_CHUNK_SIZE
        )
        audio_sink_size = (
            ECHO_WM_FLASH_AUDIO_PREFIX
            + ((block_state.video_sink_size - 1) // ECHO_WM_FLASH_VIDEO_CHUNK_SIZE) * ECHO_WM_FLASH_AUDIO_CHUNK_SIZE
        )
        caches = transformer.init_echo_wm_causal_caches(
            video_local_tokens=block_state.video_cache_size * patches_per_frame,
            video_sink_tokens=block_state.video_sink_size * patches_per_frame,
            audio_local_tokens=audio_cache_size,
            audio_sink_tokens=audio_sink_size,
        )
        video_cache_tokens = block_state.video_cache_size * patches_per_frame
        video_local_rotary_emb = transformer.rope(video_coords[:, :, :video_cache_tokens], device=device)
        audio_local_rotary_emb = transformer.audio_rope(audio_coords[:, :, :audio_cache_size], device=device)
        video_cross_rotary_emb = transformer.cross_attn_rope(video_coords[:, 0:1, :video_cache_tokens], device=device)
        audio_cross_rotary_emb = transformer.cross_attn_audio_rope(
            audio_coords[:, 0:1, :audio_cache_size], device=device
        )
        audio_to_video_slices = {}
        video_to_audio_slices = {}
        for (video_start, video_end), (audio_start, audio_end) in zip(video_blocks, audio_blocks):
            video_query_end = min(video_end, block_state.video_cache_size) * patches_per_frame
            audio_to_video_slices[(audio_start, audio_end)] = (
                video_query_end - (video_end - video_start) * patches_per_frame,
                video_query_end,
            )
            audio_query_end = min(audio_end, audio_cache_size)
            video_to_audio_slices[(video_start * patches_per_frame, video_end * patches_per_frame)] = (
                audio_query_end - (audio_end - audio_start),
                audio_query_end,
            )
        for layer_cache in caches:
            layer_cache["video_self"]["local_rotary_emb"] = video_local_rotary_emb
            layer_cache["audio_self"]["local_rotary_emb"] = audio_local_rotary_emb
            layer_cache["a2v"].update(
                local_query_rotary_emb=video_cross_rotary_emb,
                local_key_rotary_emb=audio_cross_rotary_emb,
                local_query_slices=audio_to_video_slices,
            )
            layer_cache["v2a"].update(
                local_query_rotary_emb=audio_cross_rotary_emb,
                local_key_rotary_emb=video_cross_rotary_emb,
                local_query_slices=video_to_audio_slices,
            )
        block_state.video_blocks = video_blocks
        block_state.audio_blocks = audio_blocks
        block_state.patches_per_frame = patches_per_frame
        block_state.device = device
        block_state.prompt = prompt
        block_state.audio_prompt = audio_prompt
        block_state.prompt_mask = prompt_mask
        block_state.clean_image = clean_image
        block_state.initial_video = initial_video
        block_state.initial_audio = initial_audio
        block_state.video_output = video_output
        block_state.audio_output = audio_output
        block_state.video_coords = video_coords
        block_state.audio_coords = audio_coords
        block_state.caches = caches
        block_state.sigmas = echo_wm_flash_sigmas(block_state.timesteps, components.scheduler)
        block_state.audio_num_frames = audio_num_frames
        block_state.batch_size = batch_size
        block_state.dtype = transformer.dtype

        for block_index in range(len(video_blocks)):
            components, block_state = self.loop_step(components, block_state, block_index=block_index)

        block_state.latents = block_state.video_output
        block_state.audio_latents = block_state.audio_output
        self.set_block_state(state, block_state)
        return components, state
