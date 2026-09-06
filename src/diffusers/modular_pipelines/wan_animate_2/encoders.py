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

import math

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPVisionModel, UMT5EncoderModel

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLWan
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .video_processor import WanAnimate2VideoProcessor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# Like diffusers.modular_pipelines.wan.encoders.get_t5_prompt_embeds, but without the whitespace
# cleaning -- the reference implementation encodes the prompt exactly as given.
def get_t5_prompt_embeds(
    text_encoder: UMT5EncoderModel,
    tokenizer: AutoTokenizer,
    prompt: str | list[str],
    max_sequence_length: int,
    device: torch.device,
):
    dtype = text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    return prompt_embeds


def clip_visual_encode(image_encoder, tensor, device, dtype):
    """Encode tensor to CLIP features (bicubic to 224×224, matching original)."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    videos = F.interpolate(tensor.transpose(0, 1), size=(224, 224), mode="bicubic", align_corners=False)
    videos = videos.mul_(0.5).add_(0.5)
    mean = torch.tensor(CLIP_MEAN, device=device, dtype=videos.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=videos.dtype).view(1, 3, 1, 1)
    videos = (videos - mean) / std
    out = image_encoder(pixel_values=videos.to(dtype), output_hidden_states=True)
    return out.hidden_states[-2]


def get_i2v_mask(lat_t, lat_h, lat_w, mask_len=1, device="cuda"):
    """Create an i2v mask in latent space.

    mask_len is in PIXEL space. Returns [4, lat_t, lat_h, lat_w] (no batch dim).
    """
    msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
    msk[:, :mask_len] = 1
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk


def get_frame_indices(num_frames, video_fps, target_fps):
    """Nearest-neighbour resample of a `video_fps` clip to `target_fps`."""
    num_target_frames = int(num_frames / video_fps * target_fps)
    times = np.arange(0, num_target_frames) / target_fps
    frame_indices = np.round(times * video_fps).astype(int)
    return np.clip(frame_indices, 0, num_frames - 1).tolist()


def encode_vae(vae: AutoencoderKLWan, video: torch.Tensor) -> torch.Tensor:
    """VAE-encode a `[B, C, T, H, W]` clip (mode of the distribution) and standardize the latents."""
    latents = vae.encode(video.to(vae.dtype)).latent_dist.mode()
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_recip_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    return (latents - latents_mean) * latents_recip_std


# ========================================
# Text Encoder
# ========================================


class WanAnimate2TextEncoderStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Text Encoder step that encodes the character/background prompt, the negative prompt (when the "
            "pipeline's guider needs unconditional embeddings, or one is passed explicitly), and the fixed "
            "reference prompt for the driving-video context"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", UMT5EncoderModel),
            ComponentSpec("tokenizer", AutoTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam(
                "prompt_ref",
                default="人物动作的参考视频",
                type_hint=str,
                description="The reference prompt for the driving video context",
            ),
            InputParam.template("max_sequence_length"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam(
                "prompt_ref_embeds",
                type_hint=torch.Tensor,
                description="text embeddings of the reference prompt, conditioning the reference-extraction pass",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if not isinstance(block_state.prompt, str):
            raise ValueError(f"`prompt` has to be of type `str` but is {type(block_state.prompt)}")

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        device = components._execution_device

        block_state.prompt_embeds = get_t5_prompt_embeds(
            components.text_encoder,
            components.tokenizer,
            block_state.prompt,
            block_state.max_sequence_length,
            device,
        )
        # The guider is not a component of this block: when the step runs inside the full pipeline,
        # the denoise step's guider determines (via `requires_unconditional_embeds`) whether
        # unconditional embeddings are needed, defaulting the negative prompt to "". Standalone,
        # there is no guider and the negative prompt is only encoded when the caller passes one.
        block_state.negative_prompt_embeds = None
        if components.requires_unconditional_embeds or block_state.negative_prompt is not None:
            block_state.negative_prompt_embeds = get_t5_prompt_embeds(
                components.text_encoder,
                components.tokenizer,
                block_state.negative_prompt or "",
                block_state.max_sequence_length,
                device,
            )
        block_state.prompt_ref_embeds = get_t5_prompt_embeds(
            components.text_encoder,
            components.tokenizer,
            block_state.prompt_ref,
            block_state.max_sequence_length,
            device,
        )

        self.set_block_state(state, block_state)
        return components, state


# ========================================
# Preprocessing
# ========================================


class WanAnimate2ProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Image Resize step that resolves the output frame from the target area (`height * width`) and the "
            "reference image's aspect ratio, then letterboxes the reference image into that frame. The recorded "
            "crop box is used at the end to crop the letterbox bars back off the generated video."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                WanAnimate2VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8, "spatial_patch_size": (2, 2), "resample": "bicubic"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image", description="The reference image holding the character to animate."),
            InputParam(
                "height",
                type_hint=int,
                default=800,
                description="Together with `width`, the target *area* of the generated video; the aspect ratio "
                "comes from `image`. Overwritten with the resolved frame height.",
            ),
            InputParam(
                "width",
                type_hint=int,
                default=640,
                description="See `height`. Overwritten with the resolved frame width.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_pixels",
                type_hint=torch.Tensor,
                description="The letterboxed reference image as a `[1, 3, H, W]` tensor in `[-1, 1]`",
            ),
            OutputParam(
                "crop_region",
                type_hint=tuple[int, int, int, int],
                description="`(top, left, height, width)` of the reference image content inside the letterboxed frame",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        image_height, image_width = components.image_processor.get_default_height_width(block_state.image)
        mod_value = components.vae_scale_factor_spatial * 2
        aspect_ratio = image_height / image_width
        max_area = block_state.height * block_state.width
        block_state.height = int(math.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        block_state.width = int(math.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

        height, width = block_state.height, block_state.width
        crop_width = width if width / height < image_width / image_height else image_width * height // image_height
        crop_height = height if width / height >= image_width / image_height else image_height * width // image_width
        crop_top = (height - crop_height) // 2
        crop_left = (width - crop_width) // 2
        block_state.crop_region = (crop_top, crop_left, crop_height, crop_width)

        block_state.image_pixels = components.image_processor.preprocess(
            block_state.image, height=height, width=width, resize_mode="fill"
        ).to(device, dtype=torch.float32)

        self.set_block_state(state, block_state)
        return components, state


class WanAnimate2ProcessVideosInputStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Video preprocess step that optionally resamples the driving video to the model's frame rate, "
            "letterboxes every frame into the resolved output frame, and zigzag-pads the tail so the frame count "
            "splits into whole segments. The mirrored padding frames are real content to the model; the surplus "
            "generated frames are trimmed off again at the end."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "video_processor",
                WanAnimate2VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8, "spatial_patch_size": (2, 2), "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "driving_video",
                required=True,
                type_hint=list[PIL.Image.Image],
                description="The driving video that provides the motion, in any format accepted by "
                "`VideoProcessor.preprocess_video`.",
            ),
            InputParam(
                "driving_video_fps",
                type_hint=float,
                description="The frame rate `driving_video` was captured at — `load_video(..., return_fps=True)` "
                "reports it. When set, the driving frames are resampled from it to `fps`; when `None` they are "
                "used as-is.",
            ),
            InputParam("fps", type_hint=int, default=24, description="The frame rate the model generates at"),
            InputParam(
                "segment_frame_length",
                type_hint=int,
                default=81,
                description="The number of frames in each inference segment",
            ),
            InputParam(
                "prev_segment_conditioning_frames",
                type_hint=int,
                default=1,
                description="The number of conditioning frames carried over from the previous segment",
            ),
            InputParam(
                "height",
                type_hint=int,
                default=800,
                description="The height the driving frames are letterboxed to; must match the reference image's "
                "resolved height. In the assembled pipeline the image preprocess step supplies the resolved value.",
            ),
            InputParam(
                "width",
                type_hint=int,
                default=640,
                description="See `height`.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "driving_video_pixels",
                type_hint=torch.Tensor,
                description="The resampled, letterboxed, and zigzag-padded driving video, `[1, 3, T, height, width]` "
                "in `[-1, 1]`",
            ),
            OutputParam(
                "real_frame_len",
                type_hint=int,
                description="Number of driving frames before zigzag padding; the output is trimmed to it",
            ),
            OutputParam("num_segments", type_hint=int, description="Number of inference segments"),
            OutputParam(
                "effective_segment",
                type_hint=int,
                description="Frames each segment advances by (`segment_frame_length - prev_segment_conditioning_frames`)",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        # Resample the driving video to the model's frame rate
        driving_video = block_state.driving_video
        if block_state.driving_video_fps is not None:
            frame_indices = get_frame_indices(len(driving_video), block_state.driving_video_fps, block_state.fps)
            driving_video = [driving_video[i] for i in frame_indices]

        # each frame letterboxed into the target frame -> [1, 3, T, height, width]`
        driving_video = components.video_processor.preprocess_video(
            driving_video, height=block_state.height, width=block_state.width, resize_mode="fill"
        ).to(device, dtype=torch.float32)

        # Segments overlap by `prev_segment_conditioning_frames`, so each segment advances by
        # `effective_segment` new frames.
        real_frame_len = driving_video.shape[2]
        effective_segment = block_state.segment_frame_length - block_state.prev_segment_conditioning_frames
        # If the leftover frames don't fill a whole final segment, pad it with a zigzag pattern:
        # frames [0 1 2 3 4] with 3 padding frames -> [0 1 2 3 4 | 4 3 2]. The frames generated
        # for the padding are trimmed off again in the decode step.
        if real_frame_len > block_state.prev_segment_conditioning_frames:
            leftover_frames = (real_frame_len - block_state.prev_segment_conditioning_frames) % effective_segment
        else:
            leftover_frames = 0
        num_padding = effective_segment - leftover_frames if leftover_frames > 0 else 0
        target_num_frames = real_frame_len + num_padding

        if num_padding > 0:
            padding_frames = driving_video[:, :, real_frame_len - num_padding : real_frame_len].flip(2)
            driving_video = torch.cat([driving_video, padding_frames], dim=2)

        block_state.driving_video_pixels = driving_video
        block_state.real_frame_len = real_frame_len
        block_state.effective_segment = effective_segment
        block_state.num_segments = (
            target_num_frames - block_state.prev_segment_conditioning_frames + effective_segment - 1
        ) // effective_segment

        self.set_block_state(state, block_state)
        return components, state


# ========================================
# Image Encoders (CLIP)
# ========================================


class WanAnimate2ImageClipEncoderStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return "Image Encoder step that computes CLIP vision features of the letterboxed reference image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("image_encoder", CLIPVisionModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "image_pixels",
                required=True,
                type_hint=torch.Tensor,
                description="The letterboxed reference image `[1, 3, H, W]` in `[-1, 1]`, from the image preprocess step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "encoder_hidden_states_image",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="CLIP vision features of the reference image, conditioning every denoising forward",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        block_state.encoder_hidden_states_image = clip_visual_encode(
            components.image_encoder, block_state.image_pixels[0], device, components.image_encoder.dtype
        )

        self.set_block_state(state, block_state)
        return components, state


class WanAnimate2VideoClipEncoderStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Image Encoder step that computes CLIP vision features of the driving video's first frame, "
            "conditioning the per-segment reference-extraction pass"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("image_encoder", CLIPVisionModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "driving_video_pixels",
                required=True,
                type_hint=torch.Tensor,
                description="The preprocessed driving video `[1, 3, T, H, W]`, from the video preprocess step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_clip_context",
                type_hint=torch.Tensor,
                description="CLIP vision features of the driving video's first frame",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        block_state.condition_clip_context = clip_visual_encode(
            components.image_encoder, block_state.driving_video_pixels[0, :, 0], device, components.image_encoder.dtype
        )

        self.set_block_state(state, block_state)
        return components, state


# ========================================
# VAE Encoders
# ========================================


class WanAnimate2ImageVaeEncoderStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "VAE Encoder step that encodes the letterboxed reference image and stacks the i2v conditioning mask "
            "on top, producing the reference half of the conditioning tensor `reference_latents`"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "image_pixels",
                required=True,
                type_hint=torch.Tensor,
                description="The letterboxed reference image `[1, 3, H, W]` in `[-1, 1]`, from the image preprocess step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_image_latents",
                type_hint=torch.Tensor,
                description="i2v mask + reference image latents, `[20, 1, latent_height, latent_width]`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        ref_latents = encode_vae(components.vae, block_state.image_pixels.unsqueeze(2))

        height, width = block_state.image_pixels.shape[-2:]
        latent_height = height // components.vae_scale_factor_spatial
        latent_width = width // components.vae_scale_factor_spatial

        mask_ref = get_i2v_mask(1, latent_height, latent_width, 1, device=device).to(ref_latents.dtype)
        block_state.reference_image_latents = torch.cat([mask_ref, ref_latents[0]], dim=0)

        self.set_block_state(state, block_state)
        return components, state
