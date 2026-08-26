# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma4ForConditionalGeneration,
    Gemma4UnifiedForConditionalGeneration,
    GemmaTokenizer,
    GemmaTokenizerFast,
    ProcessorMixin,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import FromSingleFileMixin, LTX2LoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...models.transformers import LTX2VideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .connectors import LTX2TextConnectors
from .dfr_core import (
    EPILOGUE_KEYFRAME_STRENGTH,
    LTX2DFRCoreMixin,
    _conditioning_fps,
    retrieve_latents,
)
from .dfr_layout import LTX2DFREpilogueTile, resolve_canvas, video_tile_plan
from .duration_head import LTX2DurationHead
from .pipeline_ltx2_condition import LTX2VideoCondition
from .pipeline_output import LTX2DFRPipelineOutput
from .utils import (
    DISTILLED_SIGMA_VALUES,
    LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT,
    LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
)
from .vocoder import LTX2Vocoder, LTX2VocoderWithBWE


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import LTX2DFRPipeline
        >>> from diffusers.utils import encode_video

        >>> pipe = LTX2DFRPipeline.from_pretrained(
        ...     "Lightricks/LTX-2.5-Diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> frame_rate = 24.0
        >>> video, audio = pipe(
        ...     prompt="A tabby cat stretching in a sunlit window, dust motes drifting in the light",
        ...     height=704,
        ...     width=1216,
        ...     num_frames=121,
        ...     frame_rate=frame_rate,
        ...     output_type="np",
        ...     return_dict=False,
        ... )

        >>> encode_video(
        ...     video[0],
        ...     fps=frame_rate,
        ...     audio=audio[0].float().cpu(),
        ...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        ...     output_path="dfr_output.mp4",
        ... )
        ```
"""


class LTX2DFRPipeline(LTX2DFRCoreMixin, DiffusionPipeline, FromSingleFileMixin, LTX2LoraLoaderMixin):
    r"""
    Pipeline for one Diffusion Fidelity Rendering (DFR) denoise pass with LTX-2.5.

    A pass generates video *and* extra single-pixel-frame keyframe slots, or re-denoises supplied latents seeded from
    those slots. Callers compose stages: this pipeline at half resolution, [`LTX2LatentUpsamplePipeline`] spatially,
    this pipeline again at full resolution with the upsampled slots and an IC-LoRA reference, then
    [`LTX2DFRTemporalRefinePipeline`] for each temporal refine round. See the LTX-2 docs for the 1080p recipe.

    Slot positions come from a segment grid aligned to the VAE's temporal border (`resolve_canvas`). The canvas is
    padded to a whole number of segments; `output_type="latent"` returns that padded grid so a slot on the pad is not
    dropped. Trim with [`trim_canvas`] before VAE decode.

    Requires a transformer whose config sets `use_keyframes_abs_pos_embedding` (LTX-2.5 and later) when
    `generate_slots=True`.

    Reference: https://github.com/Lightricks/LTX-2

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        vae ([`AutoencoderKLLTX2Video`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        audio_vae ([`AutoencoderKLLTX2Audio`]):
            Audio VAE to encode and decode audio spectrograms.
        text_encoder ([`Gemma3ForConditionalGeneration`] or [`Gemma4UnifiedForConditionalGeneration`]):
            Text encoder model.
        tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`):
            Tokenizer for the text encoder.
        connectors ([`LTX2TextConnectors`]):
            Text connector stack used to adapt text encoder hidden states for the video and audio branches.
        transformer ([`LTX2VideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        vocoder ([`LTX2Vocoder`] or [`LTX2VocoderWithBWE`]):
            Vocoder to convert mel spectrograms to audio waveforms.
        processor (`ProcessorMixin`, *optional*):
            Processor used for prompt enhancement chat templating.
        prompt_enhancer ([`Gemma4ForConditionalGeneration`], *optional*):
            Dedicated prompt enhancement model (LTX-2.5).
        duration_head ([`LTX2DurationHead`], *optional*):
            Predicts `num_frames` from the prompt embeddings when `num_frames` is not supplied.
    """

    model_cpu_offload_seq = (
        "prompt_enhancer->text_encoder->connectors->duration_head->transformer->vae->audio_vae->vocoder"
    )
    _optional_components = ["processor", "prompt_enhancer", "duration_head"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        text_encoder: Gemma3ForConditionalGeneration | Gemma4UnifiedForConditionalGeneration,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast,
        connectors: LTX2TextConnectors,
        transformer: LTX2VideoTransformer3DModel,
        vocoder: LTX2Vocoder | LTX2VocoderWithBWE,
        processor: ProcessorMixin | None = None,
        prompt_enhancer: Gemma4ForConditionalGeneration | None = None,
        duration_head: LTX2DurationHead | None = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=transformer,
            vocoder=vocoder,
            scheduler=scheduler,
            processor=processor,
            prompt_enhancer=prompt_enhancer,
            duration_head=duration_head,
        )
        self._init_dfr_runtime()

    def rebuild_epilogue_keyframes(
        self,
        keyframe_latents: torch.Tensor,
        decode_timestep: float,
        decode_noise_scale: float,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Re-encode the carry keyframes at twice their resolution by way of RGB.

        These are frames the refine rounds already settled, so the epilogue is *given* them rather than asked to
        generate them. Decoding to pixels, stretching x2 with Lanczos and encoding again preserves the frame while
        landing it on the output grid, which is what lets the epilogue pin it fully clean.

        Each plane is decoded as its own one-frame clip. The VAE is causal, so a stacked decode would let neighbouring
        planes bleed into each other -- they are independent stills, not a sequence.

        Args:
            keyframe_latents (`torch.Tensor`):
                Raw `(batch_size, C, K, H, W)` carry keyframes, as a previous pass returned them.
            seed (`int`):
                Base seed; plane `i` decodes under `seed + 4000 + i`, so a plane's pixels do not depend on how many
                planes were decoded before it.

        Returns:
            `torch.Tensor`: Raw `(batch_size, C, K, 2H, 2W)` latents, ready to pass straight back in as
            `guidance_keyframe_latents`.
        """
        if keyframe_latents.ndim != 5:
            raise ValueError(f"Expected carry keyframes (B, C, K, H, W), got {tuple(keyframe_latents.shape)}")

        keyframe_latents = self._maybe_normalize_video_latents(keyframe_latents, False)

        encoded = []
        for index in range(keyframe_latents.shape[2]):
            plane = self._denormalize_latents(
                keyframe_latents[:, :, index : index + 1],
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            )
            timestep = None
            if self.vae.config.timestep_conditioning:
                noise = randn_tensor(
                    plane.shape,
                    generator=torch.Generator(device=device).manual_seed(seed + 4000 + index),
                    device=device,
                    dtype=plane.dtype,
                )
                plane = (1 - decode_noise_scale) * plane + decode_noise_scale * noise
                timestep = torch.full((plane.shape[0],), decode_timestep, device=device, dtype=plane.dtype)
            frames = self.vae.decode(plane.to(self.vae.dtype), timestep, return_dict=False)[0]

            # PIL is the reference's Lanczos, and it wants one 8-bit channels-last image at a time.
            stretched_batch = []
            for frame in frames:
                rgb = ((frame[:, 0].float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
                image = PIL.Image.fromarray(rgb.permute(1, 2, 0).cpu().numpy(), mode="RGB")
                image = image.resize((image.width * 2, image.height * 2), resample=PIL.Image.Resampling.LANCZOS)
                stretched = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
                stretched_batch.append(stretched.permute(2, 0, 1).unsqueeze(1))
            stretched = torch.stack(stretched_batch).to(device=device, dtype=self.vae.dtype)

            encoded.append(retrieve_latents(self.vae.encode(stretched), sample_mode="argmax"))
        return torch.cat(encoded, dim=2).to(device=device, dtype=dtype)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        conditions: LTX2VideoCondition | list[LTX2VideoCondition] | None = None,
        height: int = 704,
        width: int = 1216,
        num_frames: int | None = None,
        frame_rate: float = 24.0,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        keyframes_latents: torch.Tensor | None = None,
        keyframe_positions: list[int] | None = None,
        reference_latents: torch.Tensor | None = None,
        reference_downscale_factor: int = 2,
        guidance_keyframe_latents: torch.Tensor | None = None,
        guidance_keyframe_positions: list[int] | None = None,
        guidance_keyframe_strength: float = EPILOGUE_KEYFRAME_STRENGTH,
        generate_slots: bool = True,
        sigmas: list[float] = DISTILLED_SIGMA_VALUES,
        noise_scale: float | None = None,
        freeze_audio: bool = False,
        video_tiles: list[LTX2DFREpilogueTile] | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        use_cross_timestep: bool = True,
        system_prompt: str | None = None,
        enable_prompt_enhancement: bool = False,
        prompt_max_new_tokens: int | None = None,
        prompt_enhancement_kwargs: dict[str, Any] | None = None,
        prompt_enhancement_seed: int = 10,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        One denoise pass at `height` × `width`. Compose stages in the caller: this pipeline at half-res, spatial
        upsample, this pipeline again with `latents` / `keyframes_latents` / `reference_latents`, then
        [`LTX2DFRTemporalRefinePipeline`] for each temporal round. `return_dict=False` stays `(frames, audio)` so the
        diffusion-decoder path does not break; composition uses `return_dict=True` for keyframes.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
            conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*):
                Frame-level image or video conditions. `index` is a *latent* index on this pass's `num_frames`.
            height (`int`, *optional*, defaults to `704`):
                The height in pixels of **this pass**, not the final composed output.
            width (`int`, *optional*, defaults to `1216`):
                The width in pixels of this pass.
            num_frames (`int`, *optional*):
                Pixel frame count of this pass, before internal canvas padding. If not supplied, the duration is
                predicted from the prompt by the `duration_head`. Must satisfy `(num_frames - 1) % 8 == 0`.
            frame_rate (`float`, *optional*, defaults to `24.0`):
                Playback fps of this pass. RoPE time snaps to 60 whenever this is above 30.
            min_seconds (`float`, *optional*, defaults to `1.0`):
                Lower bound on the auto-predicted duration when `num_frames` is omitted.
            max_seconds (`float`, *optional*, defaults to `20.0`):
                Upper bound on the auto-predicted duration when `num_frames` is omitted.
            latents (`torch.Tensor`, *optional*):
                Raw `(batch_size, channels, frames, height, width)` video latents to re-denoise (stage 2 / epilogue).
            audio_latents (`torch.Tensor`, *optional*):
                Raw unpacked `(batch_size, channels, length, mel_bins)` audio latents. Stage 2 still runs a joint
                audio pass; the shipped waveform of a composed recipe is stage 1's, which the caller keeps.
            keyframes_latents (`torch.Tensor`, *optional*):
                Raw `(batch_size, channels, num_slots, height, width)` slot initials, used when `generate_slots=True`.
            keyframe_positions (`list[int]`, *optional*):
                Pixel-frame indices of the generated slots. Defaults to `resolve_canvas(num_frames)`.
            reference_latents (`torch.Tensor`, *optional*):
                Raw IC-LoRA reference video (typically stage 1's frames).
            reference_downscale_factor (`int`, *optional*, defaults to `2`):
                Ratio between this pass and the reference resolution, used to scale reference token coordinates.
            guidance_keyframe_latents (`torch.Tensor`, *optional*):
                Raw pinned guidance keyframes `(batch_size, channels, K, height, width)` — the epilogue path. Not
                generated slots.
            guidance_keyframe_positions (`list[int]`, *optional*):
                Pixel-frame indices for `guidance_keyframe_latents`.
            guidance_keyframe_strength (`float`, *optional*, defaults to `1.0`):
                Conditioning strength for pinned guidance keyframes.
            generate_slots (`bool`, *optional*, defaults to `True`):
                Append generated keyframe-slot tokens. The epilogue sets this to `False` and pins
                `guidance_keyframe_latents` instead.
            sigmas (`list[float]`, *optional*):
                Noise schedule for this pass, without the terminal `0.0`.
            noise_scale (`float`, *optional*):
                Noise level unconditioned tokens start at. Defaults to `sigmas[0]`.
            freeze_audio (`bool`, *optional*, defaults to `False`):
                Hold audio at sigma 0 (epilogue / when following a frozen stage-1 waveform).
            video_tiles (`list[LTX2DFREpilogueTile]`, *optional*):
                Epilogue tiling from [`~diffusers.pipelines.ltx2.dfr_layout.epilogue_tiles`], so a resolution too
                large for one forward pass can still step a single canvas: each step runs the transformer once per
                tile and blends the predictions. Resolved into a token plan against this pass's own RoPE coordinates,
                which is why the layout is passed rather than the plan — the coordinates only exist once
                [`~LTX2DFRPipeline.prepare_latents`] has run.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator(s) for reproducibility.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                Noise scale at decode time.
            use_cross_timestep (`bool`, *optional*, defaults to `True`):
                Whether to use cross-modality sigma for cross attention modulation. `True` for LTX-2.3+.
            system_prompt (`str`, *optional*):
                Optional system prompt for prompt enhancement. See `enable_prompt_enhancement`.
            enable_prompt_enhancement (`bool`, *optional*, defaults to `False`):
                Whether to run prompt enhancement.
            prompt_max_new_tokens (`int`, *optional*):
                The maximum number of new tokens to generate when performing prompt enhancement.
            prompt_enhancement_kwargs (`dict[str, Any]`, *optional*):
                Keyword arguments for the prompt enhancer's `.generate` call.
            prompt_enhancement_seed (`int`, *optional*, defaults to `10`):
                Random seed for any random operations during prompt enhancement.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format. Choose `"pil"`, `"np"`, `"pt"` or `"latent"`. Latent output is the untrimmed canvas.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`LTX2DFRPipelineOutput`] or a plain `(frames, audio)` tuple.
            attention_kwargs (`dict`, *optional*):
                Additional kwargs passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*, defaults to `["latents"]`):
                Tensor inputs for the callback function.
            max_sequence_length (`int`, *optional*, defaults to `1024`):
                Maximum sequence length for the text prompt.

        Examples:

        Returns:
            [`LTX2DFRPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`LTX2DFRPipelineOutput`] is returned, otherwise a `tuple` of
                `(video, audio)` is returned.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            system_prompt=system_prompt,
            enable_prompt_enhancement=enable_prompt_enhancement,
            num_frames=num_frames,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            latents=latents,
            audio_latents=audio_latents,
            stg_scale=0.0,
            audio_stg_scale=0.0,
        )
        if generate_slots and not self.transformer.config.use_keyframes_abs_pos_embedding:
            raise ValueError(
                "DFR generates keyframe slots, which requires a transformer whose config sets "
                "`use_keyframes_abs_pos_embedding` (LTX-2.5 and later). Each slot costs a full latent frame of tokens, "
                "so a checkpoint without the learned marker would spend that budget on tokens it cannot interpret."
            )
        if keyframes_latents is not None and keyframes_latents.ndim != 5:
            raise ValueError(
                f"`keyframes_latents` must be unpacked 5D `(batch, channels, slots, height, width)`, got "
                f"{keyframes_latents.ndim} dims."
            )
        if guidance_keyframe_latents is not None and guidance_keyframe_latents.ndim != 5:
            raise ValueError(
                f"`guidance_keyframe_latents` must be unpacked 5D, got {guidance_keyframe_latents.ndim} dims."
            )
        if not generate_slots and keyframes_latents is not None:
            raise ValueError("`keyframes_latents` seeds generated slots; set `generate_slots=True` or omit it.")
        if guidance_keyframe_latents is not None and guidance_keyframe_positions is None:
            raise ValueError("`guidance_keyframe_positions` is required when `guidance_keyframe_latents` is passed.")

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        batch_size *= num_videos_per_prompt

        if conditions is not None and not isinstance(conditions, list):
            conditions = [conditions]

        device = self._execution_device
        noise_scale = sigmas[0] if noise_scale is None else noise_scale

        if enable_prompt_enhancement and prompt is not None:
            enhancement_image = None
            for condition in conditions or []:
                frames = condition.frames
                if isinstance(frames, PIL.Image.Image):
                    enhancement_image = frames
                    break
                if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], PIL.Image.Image):
                    enhancement_image = frames[0]
                    break
            if system_prompt is None:
                system_prompt = (
                    LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT
                    if enhancement_image is not None
                    else LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
                )
            prompt = self.enhance_prompt(
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=prompt_max_new_tokens,
                seed=prompt_enhancement_seed,
                generator=generator,
                generation_kwargs=prompt_enhancement_kwargs,
                device=device,
                image=enhancement_image,
            )

        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        video_prompt_embeds, audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=self.tokenizer_padding_side
        )

        if num_frames is None:
            if getattr(self, "duration_head", None) is None:
                raise ValueError(
                    "`num_frames` must be supplied when the pipeline has no `duration_head` component to predict it."
                )
            num_frames = self.duration_head.predict_num_frames(
                video_prompt_embeds[:1],
                audio_prompt_embeds[:1],
                frame_rate=frame_rate,
                temporal_compression_ratio=self.vae_temporal_compression_ratio,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
            )

        requested_frames = num_frames
        canvas_frames, _, resolved_positions = resolve_canvas(num_frames, self.vae_temporal_compression_ratio)
        slot_frame_indices = None
        if generate_slots:
            slot_frame_indices = (
                list(keyframe_positions) if keyframe_positions is not None else list(resolved_positions)
            )

        pinned_keyframes = None
        if guidance_keyframe_latents is not None:
            pinned_keyframes = [
                (position, guidance_keyframe_latents[:, :, index : index + 1], guidance_keyframe_strength)
                for index, position in enumerate(guidance_keyframe_positions)
            ]

        num_channels_latents = self.transformer.config.in_channels
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(canvas_frames / frame_rate * audio_latents_per_second)
        conditioning_fps = _conditioning_fps(frame_rate)

        self._num_timesteps = len(sigmas)
        progress_bar = self.progress_bar(total=self._num_timesteps)

        packed, conditioning_mask, clean_latents, video_coords, keyframes_mask, slot_token_slice = (
            self.prepare_latents(
                conditions=conditions,
                keyframe_latents=pinned_keyframes,
                slot_frame_indices=slot_frame_indices,
                slot_initial_latents=keyframes_latents,
                reference_latents=reference_latents,
                reference_downscale_factor=reference_downscale_factor,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=canvas_frames,
                frame_rate=conditioning_fps,
                noise_scale=noise_scale,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=latents,
                latents_normalized=False,
            )
        )

        resolved_tile_plan = None
        if video_tiles is not None:
            resolved_tile_plan = video_tile_plan(
                video_tiles,
                video_coords,
                (canvas_frames - 1) // self.vae_temporal_compression_ratio + 1,
                height // self.vae_spatial_compression_ratio,
                width // self.vae_spatial_compression_ratio,
            )

        if audio_latents is None:
            audio_packed = self.prepare_audio_latents(
                batch_size=batch_size,
                num_channels_latents=self.audio_latent_channels,
                audio_latent_length=audio_num_frames,
                num_mel_bins=self.audio_mel_bins,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        else:
            audio_packed = self.prepare_audio_latents(
                batch_size=batch_size,
                num_channels_latents=self.audio_latent_channels,
                audio_latent_length=audio_latents.shape[2],
                num_mel_bins=self.audio_mel_bins,
                noise_scale=0.0 if freeze_audio else noise_scale,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=audio_latents,
            )

        packed, audio_packed = self.denoise(
            latents=packed,
            conditioning_mask=conditioning_mask,
            clean_latents=clean_latents,
            video_coords=video_coords,
            keyframes_mask=keyframes_mask,
            prompt_embeds=video_prompt_embeds,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=connector_attention_mask,
            sigmas=sigmas,
            frame_rate=conditioning_fps,
            audio_latents=audio_packed,
            freeze_audio=freeze_audio,
            video_tile_plan=resolved_tile_plan,
            generator=generator,
            use_cross_timestep=use_cross_timestep,
            attention_kwargs=attention_kwargs,
            progress_bar=progress_bar,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        progress_bar.close()

        num_slots = len(slot_frame_indices) if slot_frame_indices else 0
        video_latents, slot_latents = self._unpack_video_and_slots(
            packed, canvas_frames, height, width, slot_token_slice, num_slots
        )
        public_audio = self._public_audio_from_packed(audio_packed, audio_packed.shape[1])

        out_positions = slot_frame_indices
        if slot_latents is None and pinned_keyframes is not None:
            # Epilogue: no generated slots; surface the guidance the pass was given, still normalized until finalize.
            slot_latents = self._maybe_normalize_video_latents(guidance_keyframe_latents, False)
            out_positions = list(guidance_keyframe_positions)

        return self._finalize_output(
            video_latents=video_latents,
            audio_latents=public_audio,
            keyframe_latents=slot_latents,
            keyframe_positions=out_positions,
            output_type=output_type,
            return_dict=return_dict,
            output_cls=LTX2DFRPipelineOutput,
            requested_frames=requested_frames,
            playback_fps=frame_rate,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            generator=generator,
            prompt_embeds=prompt_embeds,
        )
