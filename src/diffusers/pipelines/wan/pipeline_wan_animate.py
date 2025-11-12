# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import html
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import WanLoraLoaderMixin
from ...models import AutoencoderKLWan, WanAnimateTransformer3DModel
from ...schedulers import UniPCMultistepScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import WanAnimateImageProcessor
from .pipeline_output import WanPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> import numpy as np
        >>> from diffusers import WanAnimatePipeline
        >>> from diffusers.utils import export_to_video, load_image, load_video

        >>> model_id = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
        >>> pipe = WanAnimatePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        >>> # Optionally upcast the Wan VAE to FP32
        >>> pipe.vae.to(torch.float32)
        >>> pipe.to("cuda")

        >>> # Load the reference character image
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )

        >>> # Load pose and face videos (preprocessed from reference video)
        >>> # Note: Videos should be preprocessed to extract pose keypoints and face features
        >>> # Refer to the Wan-Animate preprocessing documentation for details
        >>> pose_video = load_video("path/to/pose_video.mp4")
        >>> face_video = load_video("path/to/face_video.mp4")

        >>> # CFG is generally not used for Wan Animate
        >>> prompt = (
        ...     "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        ...     "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        ... )

        >>> # Animation mode: Animate the character with the motion from pose/face videos
        >>> output = pipe(
        ...     image=image,
        ...     pose_video=pose_video,
        ...     face_video=face_video,
        ...     prompt=prompt,
        ...     height=height,
        ...     width=width,
        ...     segment_frame_length=77,  # Frame length of each inference segment
        ...     guidance_scale=1.0,
        ...     num_inference_steps=20,
        ...     mode="animate",
        ... ).frames[0]
        >>> export_to_video(output, "output_animation.mp4", fps=30)

        >>> # Replacement mode: Replace a character in the background video
        >>> # Requires additional background_video and mask_video inputs
        >>> background_video = load_video("path/to/background_video.mp4")
        >>> mask_video = load_video("path/to/mask_video.mp4")  # Black areas preserved, white areas generated
        >>> output = pipe(
        ...     image=image,
        ...     pose_video=pose_video,
        ...     face_video=face_video,
        ...     background_video=background_video,
        ...     mask_video=mask_video,
        ...     prompt=prompt,
        ...     height=height,
        ...     width=width,
        ...     segment_frame_length=77,  # Frame length of each inference segment
        ...     guidance_scale=1.0,
        ...     num_inference_steps=20,
        ...     mode="replace",
        ... ).frames[0]
        >>> export_to_video(output, "output_replacement.mp4", fps=30)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class WanAnimatePipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for unified character animation and replacement using Wan-Animate.

    WanAnimatePipeline takes a character image, pose video, and face video as input, and generates a video in two
    modes:

    1. **Animation mode**: The model generates a video of the character image that mimics the human motion in the input
       pose and face videos. The character is animated based on the provided motion controls, creating a new animated
       video of the character.

    2. **Replacement mode**: The model replaces a character in a background video with the provided character image,
       using the pose and face videos for motion control. This mode requires additional `background_video` and
       `mask_video` inputs. The mask video should have black regions where the original content should be preserved and
       white regions where the new character should be generated.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.WanLoraLoaderMixin.load_lora_weights`] for loading LoRA weights

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`WanAnimateTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        image_processor ([`CLIPImageProcessor`]):
            Image processor for preprocessing images before encoding.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        image_processor: CLIPImageProcessor,
        image_encoder: CLIPVisionModel,
        transformer: WanAnimateTransformer3DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.video_processor_for_mask = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial, do_normalize=False, do_convert_grayscale=True
        )
        # In case self.transformer is None (e.g. for some pipeline tests)
        spatial_patch_size = self.transformer.config.patch_size[-2:] if self.transformer is not None else (2, 2)
        self.vae_image_processor = WanAnimateImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            spatial_patch_size=spatial_patch_size,
            resample="bilinear",
            fill_color=0,
        )
        self.image_processor = image_processor

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
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

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan_i2v.WanImageToVideoPipeline.encode_image
    def encode_image(
        self,
        image: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        pose_video,
        face_video,
        background_video,
        mask_video,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        mode=None,
        prev_segment_conditioning_frames=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if pose_video is None:
            raise ValueError("Provide `pose_video`. Cannot leave `pose_video` undefined.")
        if face_video is None:
            raise ValueError("Provide `face_video`. Cannot leave `face_video` undefined.")
        if not isinstance(pose_video, list) or not isinstance(face_video, list):
            raise ValueError("`pose_video` and `face_video` must be lists of PIL images.")
        if len(pose_video) == 0 or len(face_video) == 0:
            raise ValueError("`pose_video` and `face_video` must contain at least one frame.")
        if mode == "replace" and (background_video is None or mask_video is None):
            raise ValueError(
                "Provide `background_video` and `mask_video`. Cannot leave both `background_video` and `mask_video`"
                " undefined when mode is `replace`."
            )
        if mode == "replace" and (not isinstance(background_video, list) or not isinstance(mask_video, list)):
            raise ValueError("`background_video` and `mask_video` must be lists of PIL images when mode is `replace`.")

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found"
                f" {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if mode is not None and (not isinstance(mode, str) or mode not in ("animate", "replace")):
            raise ValueError(
                f"`mode` has to be of type `str` and in ('animate', 'replace') but its type is {type(mode)} and value is {mode}"
            )

        if prev_segment_conditioning_frames is not None and (
            not isinstance(prev_segment_conditioning_frames, int) or prev_segment_conditioning_frames not in (1, 5)
        ):
            raise ValueError(
                f"`prev_segment_conditioning_frames` has to be of type `int` and 1 or 5 but its type is"
                f" {type(prev_segment_conditioning_frames)} and value is {prev_segment_conditioning_frames}"
            )

    def get_i2v_mask(
        self,
        batch_size: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        mask_len: int = 1,
        mask_pixel_values: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device] = "cuda",
    ) -> torch.Tensor:
        # mask_pixel_values shape (if supplied): [B, C = 1, T, latent_h, latent_w]
        if mask_pixel_values is None:
            mask_lat_size = torch.zeros(
                batch_size, 1, (latent_t - 1) * 4 + 1, latent_h, latent_w, dtype=dtype, device=device
            )
        else:
            mask_lat_size = mask_pixel_values.clone().to(device=device, dtype=dtype)
        mask_lat_size[:, :, :mask_len] = 1
        first_frame_mask = mask_lat_size[:, :, 0:1]
        # Repeat first frame mask self.vae_scale_factor_temporal (= 4) times in the frame dimension
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, self.vae_scale_factor_temporal, latent_h, latent_w
        ).transpose(1, 2)  # [B, C = 1, 4 * T_lat, H_lat, W_lat] --> [B, C = 4, T_lat, H_lat, W_lat]

        return mask_lat_size

    def prepare_reference_image_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        sample_mode: int = "argmax",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # image shape: (B, C, H, W) or (B, C, T, H, W)
        dtype = dtype or self.vae.dtype
        if image.ndim == 4:
            # Add a singleton frame dimension after the channels dimension
            image = image.unsqueeze(2)

        _, _, _, height, width = image.shape
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        # Encode image to latents using VAE
        image = image.to(device=device, dtype=dtype)
        if isinstance(generator, list):
            # Like in prepare_latents, assume len(generator) == batch_size
            ref_image_latents = [
                retrieve_latents(self.vae.encode(image), generator=g, sample_mode=sample_mode) for g in generator
            ]
            ref_image_latents = torch.cat(ref_image_latents)
        else:
            ref_image_latents = retrieve_latents(self.vae.encode(image), generator, sample_mode)
        # Standardize latents in preparation for Wan VAE encode
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(ref_image_latents.device, ref_image_latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            ref_image_latents.device, ref_image_latents.dtype
        )
        ref_image_latents = (ref_image_latents - latents_mean) * latents_recip_std
        # Handle the case where we supply one image and one generator, but batch_size > 1 (e.g. generating multiple
        # videos per prompt)
        if ref_image_latents.shape[0] == 1 and batch_size > 1:
            ref_image_latents = ref_image_latents.expand(batch_size, -1, -1, -1, -1)

        # Prepare I2V mask in latent space and prepend to the reference image latents along channel dim
        reference_image_mask = self.get_i2v_mask(batch_size, 1, latent_height, latent_width, 1, None, dtype, device)
        reference_image_latents = torch.cat([reference_image_mask, ref_image_latents], dim=1)

        return reference_image_latents

    def prepare_prev_segment_cond_latents(
        self,
        prev_segment_cond_video: Optional[torch.Tensor] = None,
        background_video: Optional[torch.Tensor] = None,
        mask_video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        segment_frame_length: int = 77,
        start_frame: int = 0,
        height: int = 720,
        width: int = 1280,
        prev_segment_cond_frames: int = 1,
        task: str = "animate",
        interpolation_mode: str = "bicubic",
        sample_mode: str = "argmax",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # prev_segment_cond_video shape: (B, C, T, H, W) in pixel space if supplied
        # background_video shape: (B, C, T, H, W) (same as prev_segment_cond_video shape)
        # mask_video shape: (B, 1, T, H, W) (same as prev_segment_cond_video, but with only 1 channel)
        dtype = dtype or self.vae.dtype
        if prev_segment_cond_video is None:
            if task == "replace":
                prev_segment_cond_video = background_video[:, :, :prev_segment_cond_frames].to(dtype)
            else:
                cond_frames_shape = (batch_size, 3, prev_segment_cond_frames, height, width)  # In pixel space
                prev_segment_cond_video = torch.zeros(cond_frames_shape, dtype=dtype, device=device)

        data_batch_size, channels, _, segment_height, segment_width = prev_segment_cond_video.shape
        num_latent_frames = (segment_frame_length - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        if segment_height != height or segment_width != width:
            print(
                f"Interpolating prev segment cond video from ({segment_width}, {segment_height}) to ({width}, {height})"
            )
            # Perform a 4D (spatial) rather than a 5D (spatiotemporal) reshape, following the original code
            prev_segment_cond_video = prev_segment_cond_video.transpose(1, 2).flatten(0, 1)  # [B * T, C, H, W]
            prev_segment_cond_video = F.interpolate(
                prev_segment_cond_video, size=(height, width), mode=interpolation_mode
            )
            prev_segment_cond_video = prev_segment_cond_video.unflatten(0, (batch_size, -1)).transpose(1, 2)

        # Fill the remaining part of the cond video segment with zeros (if animating) or the background video (if
        # replacing).
        if task == "replace":
            remaining_segment = background_video[:, :, prev_segment_cond_frames:].to(dtype)
        else:
            remaining_segment_frames = segment_frame_length - prev_segment_cond_frames
            remaining_segment = torch.zeros(
                batch_size, channels, remaining_segment_frames, height, width, dtype=dtype, device=device
            )

        # Prepend the conditioning frames from the previous segment to the remaining segment video in the frame dim
        prev_segment_cond_video = prev_segment_cond_video.to(dtype=dtype)
        full_segment_cond_video = torch.cat([prev_segment_cond_video, remaining_segment], dim=2)

        if isinstance(generator, list):
            if data_batch_size == len(generator):
                prev_segment_cond_latents = [
                    retrieve_latents(self.vae.encode(full_segment_cond_video[i].unsqueeze(0)), g, sample_mode)
                    for i, g in enumerate(generator)
                ]
            elif data_batch_size == 1:
                # Like prepare_latents, assume len(generator) == batch_size
                prev_segment_cond_latents = [
                    retrieve_latents(self.vae.encode(full_segment_cond_video), g, sample_mode) for g in generator
                ]
            else:
                raise ValueError(
                    f"The batch size of the prev segment video should be either {len(generator)} or 1 but is"
                    f" {data_batch_size}"
                )
            prev_segment_cond_latents = torch.cat(prev_segment_cond_latents)
        else:
            prev_segment_cond_latents = retrieve_latents(
                self.vae.encode(full_segment_cond_video), generator, sample_mode
            )
        # Standardize latents in preparation for Wan VAE encode
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(prev_segment_cond_latents.device, prev_segment_cond_latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            prev_segment_cond_latents.device, prev_segment_cond_latents.dtype
        )
        prev_segment_cond_latents = (prev_segment_cond_latents - latents_mean) * latents_recip_std

        # Prepare I2V mask
        if task == "replace":
            mask_video = 1 - mask_video
            mask_video = mask_video.permute(0, 2, 1, 3, 4)
            mask_video = mask_video.flatten(0, 1)
            mask_video = F.interpolate(mask_video, size=(latent_height, latent_width), mode="nearest")
            mask_pixel_values = mask_video.unflatten(0, (batch_size, -1))
            mask_pixel_values = mask_pixel_values.permute(0, 2, 1, 3, 4)  # output shape: [B, C = 1, T, H_lat, W_lat]
        else:
            mask_pixel_values = None
        prev_segment_cond_mask = self.get_i2v_mask(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            mask_len=prev_segment_cond_frames if start_frame > 0 else 0,
            mask_pixel_values=mask_pixel_values,
            dtype=dtype,
            device=device,
        )

        # Prepend cond I2V mask to prev segment cond latents along channel dimension
        prev_segment_cond_latents = torch.cat([prev_segment_cond_mask, prev_segment_cond_latents], dim=1)
        return prev_segment_cond_latents

    def prepare_pose_latents(
        self,
        pose_video: torch.Tensor,
        batch_size: int = 1,
        sample_mode: int = "argmax",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # pose_video shape: (B, C, T, H, W)
        pose_video = pose_video.to(device=device, dtype=dtype if dtype is not None else self.vae.dtype)
        if isinstance(generator, list):
            pose_latents = [
                retrieve_latents(self.vae.encode(pose_video), generator=g, sample_mode=sample_mode) for g in generator
            ]
            pose_latents = torch.cat(pose_latents)
        else:
            pose_latents = retrieve_latents(self.vae.encode(pose_video), generator, sample_mode)
        # Standardize latents in preparation for Wan VAE encode
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(pose_latents.device, pose_latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            pose_latents.device, pose_latents.dtype
        )
        pose_latents = (pose_latents - latents_mean) * latents_recip_std
        if pose_latents.shape[0] == 1 and batch_size > 1:
            pose_latents = pose_latents.expand(batch_size, -1, -1, -1, -1)
        return pose_latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 77,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames + 1, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    def pad_video_frames(self, frames: List[Any], num_target_frames: int) -> List[Any]:
        """
        Pads an array-like video `frames` to `num_target_frames` using a "reflect"-like strategy. The frame dimension
        is assumed to be the first dimension. In the 1D case, we can visualize this strategy as follows:

        pad_video_frames([1, 2, 3, 4, 5], 10) -> [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
        """
        idx = 0
        flip = False
        target_frames = []
        while len(target_frames) < num_target_frames:
            target_frames.append(deepcopy(frames[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(frames) - 1:
                flip = not flip

        return target_frames

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        pose_video: List[PIL.Image.Image],
        face_video: List[PIL.Image.Image],
        background_video: Optional[List[PIL.Image.Image]] = None,
        mask_video: Optional[List[PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        segment_frame_length: int = 77,
        num_inference_steps: int = 20,
        mode: str = "animate",
        prev_segment_conditioning_frames: int = 1,
        motion_encode_batch_size: Optional[int] = None,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input character image to condition the generation on. Must be an image, a list of images or a
                `torch.Tensor`.
            pose_video (`List[PIL.Image.Image]`):
                The input pose video to condition the generation on. Must be a list of PIL images.
            face_video (`List[PIL.Image.Image]`):
                The input face video to condition the generation on. Must be a list of PIL images.
            background_video (`List[PIL.Image.Image]`, *optional*):
                When mode is `"replace"`, the input background video to condition the generation on. Must be a list of
                PIL images.
            mask_video (`List[PIL.Image.Image]`, *optional*):
                When mode is `"replace"`, the input mask video to condition the generation on. Must be a list of PIL
                images.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            mode (`str`, defaults to `"animation"`):
                The mode of the generation. Choose between `"animate"` and `"replace"`.
            prev_segment_conditioning_frames (`int`, defaults to `1`):
                The number of frames from the previous video segment to be used for temporal guidance. Recommended to
                be 1 or 5. In general, should be 4N + 1, where N is a non-negative integer.
            motion_encode_batch_size (`int`, *optional*):
                The batch size for batched encoding of the face video via the motion encoder. This allows trading off
                inference speed for lower memory usage by setting a smaller batch size. Will default to
                `self.transformer.config.motion_encoder_batch_size` if not set.
            height (`int`, defaults to `720`):
                The height of the generated video.
            width (`int`, defaults to `1280`):
                The width of the generated video.
            segment_frame_length (`int`, defaults to `77`):
                The number of frames in each generated video segment. The total frames of video generated will be equal
                to the number of frames in `pose_video`; we will generate the video in segments until we have hit this
                length. In general, should be 4N + 1, where N is a non-negative integer.
            num_inference_steps (`int`, defaults to `20`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `1.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. By default, CFG is not used in Wan
                Animate inference.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            pose_video,
            face_video,
            background_video,
            mask_video,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            mode,
            prev_segment_conditioning_frames,
        )

        if segment_frame_length % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`segment_frame_length - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the"
                f" nearest number."
            )
            segment_frame_length = (
                segment_frame_length // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
        segment_frame_length = max(segment_frame_length, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # As we generate in segments of `segment_frame_length`, set the target frame length to be the least multiple
        # of the effective segment length greater than or equal to the length of `pose_video`.
        cond_video_frames = len(pose_video)
        effective_segment_length = segment_frame_length - prev_segment_conditioning_frames
        last_segment_frames = (cond_video_frames - prev_segment_conditioning_frames) % effective_segment_length
        if last_segment_frames == 0:
            num_padding_frames = 0
        else:
            num_padding_frames = effective_segment_length - last_segment_frames
        num_target_frames = cond_video_frames + num_padding_frames
        num_segments = num_target_frames // effective_segment_length

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Preprocess and encode the reference (character) image
        image_height, image_width = self.video_processor.get_default_height_width(image)
        if image_height != height or image_width != width:
            logger.warning(f"Reshaping reference image from ({image_width}, {image_height}) to ({width}, {height})")
        image_pixels = self.vae_image_processor.preprocess(image, height=height, width=width, resize_mode="fill").to(
            device, dtype=torch.float32
        )

        # Get CLIP features from the reference image
        if image_embeds is None:
            image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size * num_videos_per_prompt, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # 5. Encode conditioning videos (pose, face)
        pose_video = self.pad_video_frames(pose_video, num_target_frames)
        face_video = self.pad_video_frames(face_video, num_target_frames)

        # TODO: also support np.ndarray input (e.g. from decord like the original implementation?)
        pose_video_width, pose_video_height = pose_video[0].size
        if pose_video_height != height or pose_video_width != width:
            logger.warning(
                f"Reshaping pose video from ({pose_video_width}, {pose_video_height}) to ({width}, {height})"
            )
        pose_video = self.video_processor.preprocess_video(pose_video, height=height, width=width).to(
            device, dtype=torch.float32
        )

        face_video_width, face_video_height = face_video[0].size
        expected_face_size = self.transformer.config.motion_encoder_size
        if face_video_width != expected_face_size or face_video_height != expected_face_size:
            logger.warning(
                f"Reshaping face video from ({face_video_width}, {face_video_height}) to ({expected_face_size},"
                f" {expected_face_size})"
            )
        face_video = self.video_processor.preprocess_video(
            face_video, height=expected_face_size, width=expected_face_size
        ).to(device, dtype=torch.float32)

        if mode == "replace":
            background_video = self.pad_video_frames(background_video, num_target_frames)
            mask_video = self.pad_video_frames(mask_video, num_target_frames)

            background_video = self.video_processor.preprocess_video(background_video, height=height, width=width).to(
                device, dtype=torch.float32
            )
            mask_video = self.video_processor_for_mask.preprocess_video(mask_video, height=height, width=width).to(
                device, dtype=torch.float32
            )

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables which stay constant for all inference segments
        num_channels_latents = self.vae.config.z_dim

        # Get VAE-encoded latents of the reference (character) image
        reference_image_latents = self.prepare_reference_image_latents(
            image_pixels, batch_size * num_videos_per_prompt, generator=generator, device=device
        )

        # 8. Loop over video inference segments
        start = 0
        end = segment_frame_length  # Data space frames, not latent frames
        all_out_frames = []
        out_frames = None

        for _ in range(num_segments):
            assert start + prev_segment_conditioning_frames < cond_video_frames

            # Sample noisy latents from prior for the current inference segment
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=segment_frame_length,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=latents if start == 0 else None,  # Only use pre-calculated latents for first segment
            )

            pose_video_segment = pose_video[:, :, start:end]
            face_video_segment = face_video[:, :, start:end]

            face_video_segment = face_video_segment.expand(batch_size * num_videos_per_prompt, -1, -1, -1, -1)
            face_video_segment = face_video_segment.to(dtype=transformer_dtype)

            if start > 0:
                prev_segment_cond_video = out_frames[:, :, -prev_segment_conditioning_frames:].clone().detach()
            else:
                prev_segment_cond_video = None

            if mode == "replace":
                background_video_segment = background_video[:, :, start:end]
                mask_video_segment = mask_video[:, :, start:end]

                background_video_segment = background_video_segment.expand(
                    batch_size * num_videos_per_prompt, -1, -1, -1, -1
                )
                mask_video_segment = mask_video_segment.expand(batch_size * num_videos_per_prompt, -1, -1, -1, -1)
            else:
                background_video_segment = None
                mask_video_segment = None

            pose_latents = self.prepare_pose_latents(
                pose_video_segment, batch_size * num_videos_per_prompt, generator=generator, device=device
            )
            pose_latents = pose_latents.to(dtype=transformer_dtype)

            prev_segment_cond_latents = self.prepare_prev_segment_cond_latents(
                prev_segment_cond_video,
                background_video=background_video_segment,
                mask_video=mask_video_segment,
                batch_size=batch_size * num_videos_per_prompt,
                segment_frame_length=segment_frame_length,
                start_frame=start,
                height=height,
                width=width,
                prev_segment_cond_frames=prev_segment_conditioning_frames,
                task=mode,
                generator=generator,
                device=device,
            )

            # Concatenate the reference latents in the frame dimension
            reference_latents = torch.cat([reference_image_latents, prev_segment_cond_latents], dim=2)

            # 8.1 Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    # Concatenate the reference image + prev segment conditioning in the channel dim
                    latent_model_input = torch.cat([latents, reference_latents], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            pose_hidden_states=pose_latents,
                            face_pixel_values=face_video_segment,
                            motion_encode_batch_size=motion_encode_batch_size,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    if self.do_classifier_free_guidance:
                        # Blank out face for unconditional guidance (set all pixels to -1)
                        face_pixel_values_uncond = face_video_segment * 0 - 1
                        with self.transformer.cache_context("uncond"):
                            noise_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_hidden_states_image=image_embeds,
                                pose_hidden_states=pose_latents,
                                face_pixel_values=face_pixel_values_uncond,
                                motion_encode_batch_size=motion_encode_batch_size,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

            latents = latents.to(self.vae.dtype)
            # Destandardize latents in preparation for Wan VAE decoding
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_recip_std + latents_mean
            # Skip the first latent frame (used for conditioning)
            out_frames = self.vae.decode(latents[:, :, 1:], return_dict=False)[0]

            if start > 0:
                out_frames = out_frames[:, :, prev_segment_conditioning_frames:]
            all_out_frames.append(out_frames)

            start += effective_segment_length
            end += effective_segment_length

            # Reset scheduler timesteps / state for next denoising loop
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        self._current_timestep = None
        assert start + prev_segment_conditioning_frames >= cond_video_frames

        if not output_type == "latent":
            video = torch.cat(all_out_frames, dim=2)[:, :, :cond_video_frames]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
