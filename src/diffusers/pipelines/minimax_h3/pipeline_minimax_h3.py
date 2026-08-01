# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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
import torch
from PIL import ImageOps
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel
from ...models.autoencoders.vae import DiagonalGaussianDistribution
from ...schedulers import MiniMaxH3Scheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    prepare_keyframe_image,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .pipeline_output import MiniMaxH3PipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import MiniMaxH3Pipeline
        >>> from diffusers.utils import export_to_video, load_image
        >>> from diffusers.utils.export_utils import encode_video

        >>> # One repository holds both checkpoint partitions: this pipeline loads `transformer/`, while
        >>> # `MiniMaxH3Ref2VAPipeline.from_pretrained` on the same repository loads `transformer_ref/`.
        >>> pipe = MiniMaxH3Pipeline.from_pretrained("MiniMaxAI/MiniMax-H3", torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"

        >>> # Text to video + audio.
        >>> output = pipe(prompt=prompt, generator=torch.Generator().manual_seed(42))

        >>> # First frame (and optionally last frame) to video + audio. The canvas follows the first keyframe.
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )
        >>> output = pipe(prompt=prompt, image=image, generator=torch.Generator().manual_seed(42))

        >>> # Video and audio are generated jointly, but muxing them into one file is up to the caller.
        >>> export_to_video(output.frames[0], "output.mp4", fps=24)
        >>> encode_video(
        ...     output.frames[0], fps=24, output_path="output_with_audio.mp4",
        ...     audio=output.audio[0], audio_sample_rate=pipe.audio_sampling_rate,
        ... )
        ```
"""


class MiniMaxH3Pipeline(DiffusionPipeline):
    r"""
    Pipeline for joint video + audio generation with MiniMax-H3, covering the `t2va` (text only) and `fl2va` (first
    and/or last keyframe) tasks of the FL2VA checkpoint.

    MiniMax-H3 denoises **one packed sequence** that holds the text conditioning, the keyframe conditioning latents,
    the audio latents and the video latents at once, with a full self-attention stack over all of it. Video and audio
    step down two different schedules (`shift = 12.0` and `shift = 3.0`) inside a single transformer call per step,
    which is why the pipeline registers two schedulers.

    The checkpoint is guidance-distilled: `guidance_scale` is fixed at 1.0, there is no negative prompt, and every
    step runs exactly one forward pass.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`MiniMaxH3Transformer3DModel`]):
            The packed-sequence transformer that denoises the video and audio latents jointly.
        vae ([`AutoencoderKLMiniMaxH3`]):
            Video VAE, 16x spatially and 4x temporally. Also encodes the `fl2va` keyframes.
        audio_vae ([`AutoencoderKLMiniMaxH3Audio`]):
            Audio VAE, a waveform autoencoder at 32 kHz with 40 latents per second.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            The conditioner. MiniMax-H3 reads the hidden state after its 50th decoder layer, so the released 64-layer
            checkpoint is used with its language-model head unused; a checkpoint truncated below 51 layers is
            rejected, since the last hidden state of a truncated stack is post-norm and roughly an order of magnitude
            off in scale.
        tokenizer ([`Qwen2TokenizerFast`]):
            Tokenizer of the conditioner.
        processor ([`Qwen3VLProcessor`]):
            Processor of the conditioner, used to turn the `fl2va` keyframes into vision patches.
        scheduler ([`MiniMaxH3Scheduler`]):
            The video-latent schedule (`shift = 12.0` in the released checkpoint).
        audio_scheduler ([`MiniMaxH3Scheduler`]):
            The audio-latent schedule (`shift = 3.0` in the released checkpoint).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae->audio_vae"
    _callback_tensor_inputs = ["latents", "audio_latents", "prompt_embeds"]

    def __init__(
        self,
        transformer: MiniMaxH3Transformer3DModel,
        vae: AutoencoderKLMiniMaxH3,
        audio_vae: AutoencoderKLMiniMaxH3Audio,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: Qwen2TokenizerFast,
        processor: Qwen3VLProcessor,
        scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 16
        )
        self.vae_latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) is not None else 24
        self.audio_sampling_rate = (
            self.audio_vae.config.sampling_rate if getattr(self, "audio_vae", None) is not None else 32000
        )
        self.audio_latent_channels = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 32
        )
        self.patch_size = (
            tuple(self.transformer.config.patch_size) if getattr(self, "transformer", None) is not None else (1, 2, 2)
        )
        # The video VAE decodes into ImageNet-normalized RGB over a [0, 1] base range, which the pipeline reverts
        # itself, so the processor must not denormalize a second time.
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, do_normalize=False)

    def check_inputs(self, prompt, prompt_embeds, text_token_tags, height, width, num_frames):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Pass one of `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(prompt)}."
            )
        if (prompt_embeds is None) != (text_token_tags is None):
            raise ValueError("`prompt_embeds` and `text_token_tags` have to be passed together.")
        if prompt_embeds is not None and prompt_embeds.shape[1] != text_token_tags.shape[0]:
            raise ValueError(
                f"`text_token_tags` must hold one tag per row of `prompt_embeds`, got {text_token_tags.shape[0]} tags "
                f"for {prompt_embeds.shape[1]} rows."
            )
        if (height is None) != (width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if height is not None and (height % MINIMAX_H3_CANVAS_MULTIPLE or width % MINIMAX_H3_CANVAS_MULTIPLE):
            raise ValueError(
                f"`height` and `width` must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}, got {height}x{width}."
            )
        # The duration the request generates is the one of the *aligned* frame count, so that is what the ceiling has
        # to hold for: 346 frames would otherwise pass the check and then be rounded up to 362, i.e. 15.083 seconds.
        aligned_num_frames = align_num_frames(num_frames)
        duration = aligned_num_frames / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 generates between {MINIMAX_H3_MIN_DURATION} and {MINIMAX_H3_MAX_DURATION} seconds at "
                f"{MINIMAX_H3_FPS} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS)} and "
                f"{int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)}, got {num_frames} (rounded up to "
                f"{aligned_num_frames})."
            )

    def encode_prompt(
        self,
        prompt: str,
        images: list | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build MiniMax-H3's presentation of a request and encode it.

        The presentation is the verbatim prompt for `t2va`. Every keyframe prepends a `"<Picture i>: "` label and a
        vision block (`<|vision_start|>`, one `<|image_pad|>` per vision patch, `<|vision_end|>`) — no chat template
        and no special tokens. The rows of a vision block are tagged as *video* rather than text, which is what the
        transformer's AdaLN modulation keys off.

        Args:
            prompt (`str`): The prompt to encode.
            images (`list[PIL.Image.Image]`, *optional*):
                The keyframes, already prepared onto the target canvas, in packed order.
            device (`torch.device`, *optional*): The device to run the conditioner on.
            dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(1, num_text_tokens, 5120)` hidden states and the
            `(num_text_tokens,)` per-row modality tags.
        """
        device = device or self._execution_device
        dtype = dtype or self.transformer.dtype

        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                f"MiniMax-H3 conditions on `hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}]` of its Qwen3-VL "
                f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
                f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
                f"{MINIMAX_H3_TEXT_ENCODER_LAYER} layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        pixel_values, image_grid_thw = None, None
        token_ids, token_tags = [], []
        if images:
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge_size = self.processor.image_processor.merge_size**2
            for index in range(len(images)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        # Qwen3-VL lays its 3D rotary positions out per modality run, which it reads off the token type ids the
        # processor derives from the vision pad ids (`0` text, `1` image, `2` video).
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
        )
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None if pixel_values is None else pixel_values.to(device, self.text_encoder.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
            use_cache=False,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(device=device, dtype=dtype)
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)

    def encode_keyframes(self, images: list, device: torch.device | None = None) -> torch.Tensor:
        r"""
        Encode the `fl2va` keyframes into packed conditioning rows.

        The keyframes go through the video VAE's spatial encoder only — they are single frames, so none of its
        17-frame temporal chunking applies — and the posterior is *sampled*, under a generator seeded with 42
        independently of the request seed. The sampled latent is rounded to float16 before being normalized, as in the
        reference implementation; both are part of reproducing the released model's conditioning.

        Args:
            images (`list[PIL.Image.Image]`):
                The keyframes, already prepared onto the target canvas, in packed order.
            device (`torch.device`, *optional*): The device to run the VAE on.

        Returns:
            `torch.Tensor` of shape `(num_condition_rows, latent_channels * prod(patch_size))`: the float32
            conditioning rows.
        """
        device = device or self._execution_device
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

        rows = []
        for image in images:
            pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            # `vae.encode` chunks along time for videos; a keyframe is one frame and is encoded by the (tiled)
            # spatial encoder alone, which is what the released model conditions on.
            moments = self.vae._encode_clip(pixels)
            posterior = DiagonalGaussianDistribution(moments)
            latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
            # The sampled latent is rounded to float16 before it is normalized: ~11 bits of every conditioning
            # latent, so the released model's conditioning cannot be reproduced without it.
            latents = latents.to(torch.float16).float().cpu()
            rows.append(patchify_video_latents((latents - latents_mean) / latents_std, self.patch_size))
        return torch.cat(rows)

    def prepare_latents(
        self,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Draw the initial noise of both modalities and pack it into transformer rows.

        A request draws every stream from the one generator it is given, and the order is part of what that generator
        reproduces: the conditioning noise of the keyframes or references first (one draw per condition, in
        [`~pipelines.minimax_h3.packing.keyframe_condition_noise`]), then the video noise here, as a latent tensor
        that is patchified afterwards, then the audio noise, directly in row layout. Passing `latents` or
        `audio_latents` skips its draw and shifts the ones after it.

        Args:
            num_latent_frames (`int`): Number of video latent frames.
            latent_height (`int`): Latent height.
            latent_width (`int`): Latent width.
            num_audio_latents (`int`): Number of audio latents per channel.
            device (`torch.device`): The device the rows are drawn on.
            generator (`torch.Generator`, *optional*): The generator of the request.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video noise of shape `(1, latent_channels, num_latent_frames, latent_height,
                latent_width)`, used instead of the draw.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio noise of shape `(2, audio_latent_channels, num_audio_latents)`.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the video rows and the channel-major audio rows.
        """
        if latents is None:
            latents = randn_tensor(
                (1, self.vae_latent_channels, num_latent_frames, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(latents.to(torch.float32), self.patch_size)

        if audio_latents is None:
            audio_rows = randn_tensor(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, self.audio_latent_channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            audio_rows = audio_latents.to(torch.float32).permute(0, 2, 1).reshape(-1, self.audio_latent_channels)
        return video_rows.to(device), audio_rows.to(device)

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | None = None,
        image=None,
        last_image=None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int = 124,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        text_token_tags: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents", "audio_latents"],
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str`, *optional*):
                The prompt to guide generation. MiniMax-H3 packs one request into one sequence, so only a single
                prompt is accepted.
            image (`PIL.Image.Image`, *optional*):
                Keyframe the video starts from. It is *stretched* onto the target canvas, which by default is derived
                from its own aspect ratio.
            last_image (`PIL.Image.Image`, *optional*):
                Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. Combined with
                `image` it is the follower of the two and is cover-cropped onto the canvas.
            height (`int`, *optional*):
                Height of the generated video, a multiple of 32. Defaults, together with `width`, to MiniMax-H3's own
                768-short-edge canvas for the aspect ratio of `image` (or 16:9 without keyframes).
            width (`int`, *optional*):
                Width of the generated video, a multiple of 32.
            num_frames (`int`, defaults to `124`):
                Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the VAE can
                decode; the resulting duration must stay between 5 and 15 seconds.
            num_inference_steps (`int`, defaults to `50`):
                Number of denoising steps.
            generator (`torch.Generator`, *optional*):
                A generator to make generation deterministic. Every noise draw of a request — the keyframe
                conditioning noise, the video noise and the audio noise, in that order — comes off this one
                generator, so the same generator state reproduces a sample.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video noise, of shape `(1, 24, num_latent_frames, height // 16, width // 16)`.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio noise, of shape `(2, 32, num_audio_latents)`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed conditioning of shape `(1, num_text_tokens, 5120)`, as returned by
                [`~MiniMaxH3Pipeline.encode_prompt`], which skips the conditioner. Must be passed together with
                `text_token_tags`, and for `fl2va` must have been encoded with the very keyframes passed as `image` /
                `last_image` — those are still needed here, for their conditioning latents.
            text_token_tags (`torch.Tensor`, *optional*):
                The per-row modality tags that go with `prompt_embeds`.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generated video: `"pil"`, `"np"`, `"pt"` or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~pipelines.minimax_h3.MiniMaxH3PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Passed to the attention processors, and may carry a `scale` entry for the LoRA layers.
            callback_on_step_end (`Callable`, *optional*):
                Called at the end of every denoising step with `(self, step, timestep, callback_kwargs)`. The tensors
                it receives are named by `callback_on_step_end_tensor_inputs`; note that `latents` and `audio_latents`
                are the packed rows of the whole sequence during the loop.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents", "audio_latents"]`):
                Which local tensors to pass to `callback_on_step_end`.

        Examples:

        Returns:
            [`~pipelines.minimax_h3.MiniMaxH3PipelineOutput`] or `tuple`:
                The generated video and its soundtrack.
        """
        for name in callback_on_step_end_tensor_inputs:
            if name not in self._callback_tensor_inputs:
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, got {name}."
                )

        # 1. Resolve the geometry. MiniMax-H3 was released at a 768 pixel short edge with a 768x1344 area cap, and
        # the video VAE only encodes 17 * n + 5 frames.
        self.check_inputs(prompt, prompt_embeds, text_token_tags, height, width, num_frames)
        keyframes = [
            ImageOps.exif_transpose(keyframe).convert("RGB")
            for keyframe in (image, last_image)
            if keyframe is not None
        ]
        keyframe_anchors = tuple(
            anchor for anchor, keyframe in (("first", image), ("last", last_image)) if keyframe is not None
        )
        if height is None:
            height, width = resolve_canvas_size(*(keyframes[0].size if keyframes else (16, 9)))
        aligned_num_frames = align_num_frames(num_frames)
        if aligned_num_frames != num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {num_frames} up to "
                f"{aligned_num_frames}."
            )
            num_frames = aligned_num_frames

        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        num_latent_frames = video_latent_num_frames(num_frames)
        num_audio_latents = audio_latent_num_frames(num_frames)

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        device = self._execution_device

        # 2. Encode the prompt, including a vision block per keyframe.
        keyframes = [
            prepare_keyframe_image(keyframe, height, width, stretch=index == 0)
            for index, keyframe in enumerate(keyframes)
        ]
        if prompt_embeds is None:
            prompt_embeds, text_token_tags = self.encode_prompt(prompt, keyframes, device=device)

        # 3. Build the packed layout: [text | keyframe conditions | target audio | target video].
        layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
            keyframe_anchors,
        )
        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)

        # 4. Encode the keyframes and noise them to their conditioning level. The anchors are fixed for the whole
        # loop: they are re-imposed by construction, since the loop only ever writes the target rows.
        condition_rows = None
        if keyframes:
            condition_rows = self.encode_keyframes(keyframes, device=device)
            noise = keyframe_condition_noise(
                ((1, latent_height, latent_width),) * len(keyframes),
                self.patch_size,
                self.vae_latent_channels,
                generator=generator,
                device=device,
            )
            condition_rows = self.scheduler.scale_noise(
                condition_rows.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise
            )

        # 5. The two schedules, and the noise.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.audio_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, audio_timesteps = self.scheduler.timesteps, self.audio_scheduler.timesteps
        self._num_timesteps = len(timesteps)

        latents, audio_latents = self.prepare_latents(
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            device,
            generator,
            latents,
            audio_latents,
        )
        if condition_rows is not None:
            latents = torch.cat([condition_rows, latents])
        num_condition_rows = layout.num_condition_video_rows

        # 6. Denoise. One forward per step serves every modality and every noise level at once: the conditioning rows
        # ride along at their own timestep and are never updated. The row-to-timestep assignment is static per step,
        # so the whole plan is staged before the loop rather than rebuilt on every one of them.
        timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in build_row_timesteps(
                    layout, float(t), float(audio_t), max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG), 1.0
                )
            )
            for t, audio_t in zip(timesteps, audio_timesteps)
        ]

        with self.progress_bar(total=self._num_timesteps) as progress_bar:
            for i, (t, audio_t) in enumerate(zip(timesteps, audio_timesteps)):
                if self.interrupt:
                    continue

                unique_timesteps, timestep_indices = timestep_plan[i]
                noise_pred, audio_noise_pred = self.transformer(
                    hidden_states=latents[None],
                    audio_hidden_states=audio_latents[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps,
                    timestep_indices=timestep_indices,
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=video_indices,
                    audio_indices=audio_indices,
                    text_indices=text_indices,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )

                latents[num_condition_rows:] = self.scheduler.step(
                    noise_pred[0, num_condition_rows:].float(),
                    t,
                    latents[num_condition_rows:],
                    return_dict=False,
                )[0]
                audio_latents = self.audio_scheduler.step(
                    audio_noise_pred[0].float(), audio_t, audio_latents, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[name] = locals()[name]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    audio_latents = callback_outputs.pop("audio_latents", audio_latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

        # 7. Unpack the generated rows. The spatial tiling of the VAE covers the canvas exactly (tiles and overlaps
        # are both latent-aligned), so the decoded frames need no crop back to `latent_height * 16`.
        latents = unpatchify_video_tokens(
            latents[num_condition_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.vae_latent_channels,
            self.patch_size,
        )
        audio_latents = unpack_audio_tokens(audio_latents, num_audio_latents)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        audio_latents_mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        audio_latents_std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        latents = latents * latents_std + latents_mean
        audio_latents = audio_latents * audio_latents_std + audio_latents_mean

        if output_type == "latent":
            video, audio = latents, audio_latents
        else:
            # MiniMax-H3 decodes video under float16 autocast even though the VAE weights are float32.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                video = self.vae.decode(latents, return_dict=False)[0]
            pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            # The audio VAE is mono and takes the two stereo channels as two batch items.
            audio = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = audio.float().permute(1, 0, 2)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, audio)
        return MiniMaxH3PipelineOutput(frames=video, audio=audio)
