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

import torch
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma4UnifiedForConditionalGeneration,
    GemmaTokenizer,
    GemmaTokenizerFast,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import FromSingleFileMixin, LTX2LoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...models.transformers import LTX2VideoTransformer3DModel
from ...schedulers import LTXEulerAncestralRFScheduler
from ...utils import replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from .connectors import LTX2TextConnectors
from .dfr_core import (
    ANCHOR_KEYFRAME_STRENGTH,
    TEMPORAL_ANCESTRAL_ETA,
    LTX2DFRCoreMixin,
    _audio_window_for_tile,
    _conditioning_fps,
)
from .dfr_layout import temporal_tile_plan
from .latent_upsampler import LTX2LatentUpsamplerModel
from .pipeline_ltx2_condition import LTX2VideoCondition
from .pipeline_output import LTX2DFRPipelineOutput
from .utils import TEMPORAL_ROUND_DISTILLED_SIGMA_VALUES
from .vocoder import LTX2Vocoder, LTX2VocoderWithBWE


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import LTX2DFRPipeline, LTX2DFRTemporalRefinePipeline, LTXEulerAncestralRFScheduler
        >>> from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel

        >>> pipe = LTX2DFRPipeline.from_pretrained("Lightricks/LTX-2.5-Diffusers", torch_dtype=torch.bfloat16)
        >>> temporal_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        ...     "path/to/converted/temporal_latent_upsampler", torch_dtype=torch.bfloat16
        ... )
        >>> temporal_pipe = LTX2DFRTemporalRefinePipeline(
        ...     scheduler=LTXEulerAncestralRFScheduler(eta=0.5),
        ...     vae=pipe.vae,
        ...     audio_vae=pipe.audio_vae,
        ...     text_encoder=pipe.text_encoder,
        ...     tokenizer=pipe.tokenizer,
        ...     connectors=pipe.connectors,
        ...     transformer=pipe.transformer,
        ...     vocoder=pipe.vocoder,
        ...     temporal_latent_upsampler=temporal_upsampler,
        ... )
        >>> # `out` is a prior DFR pass at the same spatial size, `return_dict=True`.
        >>> out = temporal_pipe(
        ...     latents=out.frames,
        ...     keyframes_latents=out.keyframes,
        ...     keyframe_positions=out.keyframe_positions,
        ...     audio_latents=out.audio,
        ...     prompt="A tabby cat stretching in a sunlit window",
        ...     height=1088,
        ...     width=1920,
        ...     num_frames=121,
        ...     output_type="latent",
        ... )
        ```
"""


class LTX2DFRTemporalRefinePipeline(LTX2DFRCoreMixin, DiffusionPipeline, FromSingleFileMixin, LTX2LoraLoaderMixin):
    r"""
    One temporal DFR refine round: interpolate the canvas x2 in time, tile on keyframe seams, ancestral-denoise each
    tile, stitch by dropping the later tile's lead-in, and merge the carry-keyframe bag.

    The scheduler is [`LTXEulerAncestralRFScheduler`] (`eta=0.5` by default). Stage 1 / 2 / the spatial epilogue stay
    on [`FlowMatchEulerDiscreteScheduler`] via [`LTX2DFRPipeline`]. Call this pipeline once per round; the caller
    loops for 2x / 4x.

    Incoming `latents` / `keyframes_latents` / `audio_latents` are raw (denormalized), matching [`LTX2Pipeline`].
    `keyframe_positions` must be the positions returned by the previous pass — they cannot be re-derived from the
    original `num_frames` after a round has run.

    Args:
        scheduler ([`LTXEulerAncestralRFScheduler`]):
            Ancestral Euler scheduler in the rectified-flow parameterization. Construct with `eta=0.5` to match the
            DFR temporal recipe.
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
        temporal_latent_upsampler ([`LTX2LatentUpsamplerModel`]):
            Temporal x2 latent upsampler applied at the start of the round.
    """

    model_cpu_offload_seq = "text_encoder->connectors->transformer->temporal_latent_upsampler->vae->audio_vae->vocoder"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: LTXEulerAncestralRFScheduler,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        text_encoder: Gemma3ForConditionalGeneration | Gemma4UnifiedForConditionalGeneration,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast,
        connectors: LTX2TextConnectors,
        transformer: LTX2VideoTransformer3DModel,
        vocoder: LTX2Vocoder | LTX2VocoderWithBWE,
        temporal_latent_upsampler: LTX2LatentUpsamplerModel,
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
            temporal_latent_upsampler=temporal_latent_upsampler,
        )
        self._init_dfr_runtime()

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        latents: torch.Tensor = None,
        keyframes_latents: torch.Tensor = None,
        keyframe_positions: list[int] = None,
        audio_latents: torch.Tensor = None,
        conditions: LTX2VideoCondition | list[LTX2VideoCondition] | None = None,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        source_seconds: float | None = None,
        condition_num_frames: int | None = None,
        round_index: int = 1,
        sigmas: list[float] = TEMPORAL_ROUND_DISTILLED_SIGMA_VALUES,
        noise_scale: float | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        use_cross_timestep: bool = True,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
    ):
        r"""
        Run one temporal refine round.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
            latents (`torch.Tensor`):
                Raw video latents of the **input** canvas, `(batch_size, channels, frames, height, width)`.
            keyframes_latents (`torch.Tensor`):
                Raw carry keyframes `(batch_size, channels, K, height, width)` from the previous pass.
            keyframe_positions (`list[int]`):
                Pixel-frame indices of `keyframes_latents` on the **input** canvas.
            audio_latents (`torch.Tensor`):
                Frozen stage-1 audio, unpacked and denormalized. Each tile is handed the slice covering its playback
                window; the returned audio is this waveform, not a per-tile re-denoise.
            conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*):
                Frame-level conditions indexed on `condition_num_frames` (the original request). Each round scales a
                condition's pixel position by `2 ** round_index`.
            height (`int`, *optional*, defaults to `704`):
                Pixel height of this pass (same as the incoming video).
            width (`int`, *optional*, defaults to `1216`):
                Pixel width of this pass.
            num_frames (`int`, *optional*, defaults to `121`):
                Pixel frame count of the **input** canvas (untrimmed).
            frame_rate (`float`, *optional*, defaults to `24.0`):
                Playback fps of the **input**. The round doubles it.
            source_seconds (`float`, *optional*):
                Duration of the frozen stage-1 audio. Defaults to `num_frames / frame_rate`, which is correct for the
                first round; later rounds must pass the original stage-1 duration so tiles do not drift.
            condition_num_frames (`int`, *optional*):
                Original generation `num_frames` used to encode `conditions`. Defaults to `num_frames`. After padding
                or a prior round, pass the original request so `index=-1` does not wrap to the padded tail.
            round_index (`int`, *optional*, defaults to `1`):
                1-based round number. Tiles seed ancestral noise as `seed + 1000 * round_index + tile`, and conditions
                are scaled by `2 ** round_index`.
            sigmas (`list[float]`, *optional*):
                Distilled schedule for this round's tiles, without the terminal `0.0`.
            noise_scale (`float`, *optional*):
                Noise level unconditioned tokens start at. Defaults to `sigmas[0]`.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator(s) for reproducibility. Ancestral draws use a separate per-tile seed derived from
                this generator's `initial_seed()`.
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
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format. Choose `"pil"`, `"np"`, `"pt"` or `"latent"`. Latent output is the untrimmed canvas.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`LTX2DFRPipelineOutput`] or a plain `(frames, audio)` tuple.
            attention_kwargs (`dict`, *optional*):
                Additional kwargs passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step, across every tile.
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

        if latents is None or keyframes_latents is None or keyframe_positions is None or audio_latents is None:
            raise ValueError(
                "`latents`, `keyframes_latents`, `keyframe_positions`, and `audio_latents` are required. They come "
                "from the previous DFR pass (`return_dict=True`)."
            )
        if getattr(self, "temporal_latent_upsampler", None) is None:
            raise ValueError("LTX2DFRTemporalRefinePipeline requires the `temporal_latent_upsampler` component.")
        if not self.transformer.config.use_keyframes_abs_pos_embedding:
            raise ValueError(
                "Temporal DFR invents mid-segment keyframe slots, which requires a transformer whose config sets "
                "`use_keyframes_abs_pos_embedding` (LTX-2.5 and later)."
            )
        if round_index < 1:
            raise ValueError(f"`round_index` must be >= 1, got {round_index}")
        # A round densifies freshly interpolated frames, which needs the ancestral renoise. `denoise` takes a plain
        # Euler step for any other scheduler, so an unchecked component here would run to completion and just return
        # a softer canvas -- the one failure mode with no signal at all.
        if not isinstance(self.scheduler, LTXEulerAncestralRFScheduler):
            raise ValueError(
                f"Temporal refine needs `LTXEulerAncestralRFScheduler(eta={TEMPORAL_ANCESTRAL_ETA})`, got "
                f"{type(self.scheduler).__name__}. `from_pretrained` picks up the repo's flow-match scheduler, which "
                f"steps deterministically and silently loses the refine round's detail; construct the scheduler "
                f"explicitly."
            )
        if float(self.scheduler.config.eta) <= 0:
            raise ValueError(
                f"Temporal refine needs a stochastic step, but the scheduler has `eta={self.scheduler.config.eta}`. "
                f"Use `LTXEulerAncestralRFScheduler(eta={TEMPORAL_ANCESTRAL_ETA})`."
            )

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_frames=num_frames,
            latents=latents,
            audio_latents=audio_latents,
            stg_scale=0.0,
            audio_stg_scale=0.0,
        )

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if conditions is not None and not isinstance(conditions, list):
            conditions = [conditions]

        device = self._execution_device
        noise_scale = sigmas[0] if noise_scale is None else noise_scale
        source_seconds = num_frames / frame_rate if source_seconds is None else source_seconds
        condition_num_frames = num_frames if condition_num_frames is None else condition_num_frames
        seed_source = generator[0] if isinstance(generator, list) else generator
        ancestral_seed_base = seed_source.initial_seed() if seed_source is not None else 0

        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        video_prompt_embeds, audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=self.tokenizer_padding_side
        )

        num_channels_latents = self.transformer.config.in_channels
        temporal_ratio = self.vae_temporal_compression_ratio
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )

        video_latents = self._maybe_normalize_video_latents(latents.to(device=device), False)
        carry_keyframes = self._maybe_normalize_video_latents(keyframes_latents.to(device=device), False)
        stage_1_audio = self._pack_public_audio(audio_latents.to(device=device, dtype=torch.float32))

        video_latents = self.upsample_latents(video_latents, self.temporal_latent_upsampler)
        canvas_frames = 2 * (num_frames - 1) + 1
        playback_fps = 2 * frame_rate
        conditioning_fps = _conditioning_fps(playback_fps)
        seam_positions = [2 * position for position in keyframe_positions]
        tiles = temporal_tile_plan(seam_positions, canvas_frames, 2**round_index, temporal_ratio)
        pixel_scale = 2**round_index

        self._num_timesteps = len(sigmas) * len(tiles)
        progress_bar = self.progress_bar(total=self._num_timesteps)
        step_offset = 0

        round_conditions = self.encode_conditions(
            conditions, height, width, condition_num_frames, device=device, dtype=torch.float32
        )

        tile_latents = []
        slot_positions: list[int] = []
        slot_latents: list[torch.Tensor] = []
        seam_to_index = {seam: index for index, seam in enumerate(seam_positions)}
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        tokens_per_latent_frame = latent_height * latent_width

        for tile_index, tile in enumerate(tiles):
            tile_frames = (tile.interval.end - tile.interval.start - 1) * temporal_ratio + 1
            tile_video_latents = video_latents[:, :, tile.interval.start : tile.interval.end]

            tile_conditions = [
                (pixel * pixel_scale - tile.pixel_start, latent, strength, num_pixel_frames)
                for pixel, latent, strength, num_pixel_frames in round_conditions
                if tile.pixel_start <= pixel * pixel_scale <= tile.pixel_end
            ]

            tile_keyframe_latents = [
                (
                    position - tile.pixel_start,
                    carry_keyframes[:, :, seam_to_index[position] : seam_to_index[position] + 1],
                    ANCHOR_KEYFRAME_STRENGTH,
                )
                for position in tile.anchors
            ]

            tile_slot_positions = [position - tile.pixel_start for position in tile.slots]
            seed_indices = [
                min(round(position / temporal_ratio), tile_video_latents.shape[2] - 1)
                for position in tile_slot_positions
            ]
            tile_slot_initials = (
                torch.cat([tile_video_latents[:, :, index : index + 1] for index in seed_indices], dim=2)
                if seed_indices
                else None
            )

            (
                tile_packed_latents,
                tile_conditioning_mask,
                tile_clean_latents,
                tile_video_coords,
                tile_keyframes_mask,
                tile_slot_slice,
            ) = self.prepare_latents(
                condition_latents=tile_conditions,
                keyframe_latents=tile_keyframe_latents,
                slot_frame_indices=tile_slot_positions or None,
                slot_initial_latents=tile_slot_initials,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=tile_frames,
                frame_rate=conditioning_fps,
                noise_scale=noise_scale,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=tile_video_latents,
            )
            ancestral_generator = torch.Generator(device=device).manual_seed(
                ancestral_seed_base + 1000 * round_index + tile_index
            )
            tile_audio_latents = _audio_window_for_tile(
                stage_1_audio,
                pixel_start=tile.pixel_start,
                tile_frames=tile_frames,
                playback_fps=playback_fps,
                source_seconds=source_seconds,
                conditioning_fps=conditioning_fps,
                audio_latents_per_second=audio_latents_per_second,
            )
            tile_packed_latents, _ = self.denoise(
                latents=tile_packed_latents,
                conditioning_mask=tile_conditioning_mask,
                clean_latents=tile_clean_latents,
                video_coords=tile_video_coords,
                keyframes_mask=tile_keyframes_mask,
                prompt_embeds=video_prompt_embeds,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=connector_attention_mask,
                sigmas=sigmas,
                frame_rate=conditioning_fps,
                audio_latents=tile_audio_latents,
                freeze_audio=True,
                generator=ancestral_generator,
                use_cross_timestep=use_cross_timestep,
                attention_kwargs=attention_kwargs,
                progress_bar=progress_bar,
                step_offset=step_offset,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            )
            step_offset += len(sigmas)

            unpacked = self._unpack_video_latents(
                tile_packed_latents[:, : tile_video_latents.shape[2] * tokens_per_latent_frame],
                tile_video_latents.shape[2],
                latent_height,
                latent_width,
            )
            tile_latents.append(unpacked)
            if tile_slot_slice is not None:
                slot_positions.extend(tile.slots)
                slot_latents.append(
                    self._unpack_video_latents(
                        tile_packed_latents[:, tile_slot_slice],
                        len(tile_slot_positions),
                        latent_height,
                        latent_width,
                    )
                )

        video_latents = torch.cat(
            [latent[:, :, tile.interval.left_ramp :] for latent, tile in zip(tile_latents, tiles)], dim=2
        )
        expected_latent_frames = (canvas_frames - 1) // temporal_ratio + 1
        if video_latents.shape[2] != expected_latent_frames:
            raise RuntimeError(
                f"Stitched round {round_index} has T={video_latents.shape[2]} latent frames, expected "
                f"{expected_latent_frames}"
            )

        carry: dict[int, torch.Tensor] = {
            position: carry_keyframes[:, :, index : index + 1] for index, position in enumerate(seam_positions)
        }
        all_slot_latents = torch.cat(slot_latents, dim=2) if slot_latents else None
        first_slot_index: dict[int, int] = {}
        for index, position in enumerate(slot_positions):
            first_slot_index.setdefault(position, index)
        for position, index in first_slot_index.items():
            carry[position] = all_slot_latents[:, :, index : index + 1]
        carry_positions = sorted(carry)
        carry_keyframes = torch.cat([carry[position] for position in carry_positions], dim=2)

        progress_bar.close()

        public_audio = self._public_audio_from_packed(stage_1_audio, stage_1_audio.shape[1])
        return self._finalize_output(
            video_latents=video_latents,
            audio_latents=public_audio,
            keyframe_latents=carry_keyframes,
            keyframe_positions=carry_positions,
            output_type=output_type,
            return_dict=return_dict,
            output_cls=LTX2DFRPipelineOutput,
            requested_frames=None,
            playback_fps=playback_fps,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            generator=generator,
            prompt_embeds=prompt_embeds,
        )
