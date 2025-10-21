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
        >>> from diffusers import AutoencoderKLWan, WanAnimatePipeline
        >>> from diffusers.utils import export_to_video, load_image, load_video
        >>> from transformers import CLIPVisionModel

        >>> model_id = "Wan-AI/Wan2.2-Animate-14B-720P-Diffusers"
        >>> image_encoder = CLIPVisionModel.from_pretrained(
        ...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
        ... )
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanAnimatePipeline.from_pretrained(
        ...     model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> # Load the character image
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )

        >>> # Load pose and face videos (preprocessed from reference video)
        >>> # Note: Videos should be preprocessed to extract pose keypoints and face features
        >>> # Refer to the Wan-Animate preprocessing documentation for details
        >>> pose_video = load_video("path/to/pose_video.mp4")
        >>> face_video = load_video("path/to/face_video.mp4")

        >>> # Calculate optimal dimensions based on VAE constraints
        >>> max_area = 480 * 832
        >>> aspect_ratio = image.height / image.width
        >>> mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        >>> height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        >>> width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        >>> image = image.resize((width, height))

        >>> prompt = (
        ...     "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        ...     "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        ... )
        >>> negative_prompt = (
        ...     "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
        ...     "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        ...     "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
        ...     "messy background, three legs, many people in the background, walking backwards"
        ... )

        >>> # Animation mode: Animate the character with the motion from pose/face videos
        >>> output = pipe(
        ...     image=image,
        ...     pose_video=pose_video,
        ...     face_video=face_video,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=height,
        ...     width=width,
        ...     num_frames=77,
        ...     guidance_scale=1.0,
        ...     mode="animation",
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
        ...     negative_prompt=negative_prompt,
        ...     height=height,
        ...     width=width,
        ...     num_frames=76,
        ...     guidance_scale=1.0,
        ...     mode="replacement",
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
            vae_scale_factor=self.vae_scale_factor_spatial, do_normalize=False
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

    def encode_image(
        self,
        image: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-1]

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
        num_frames_for_temporal_guidance=None,
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
        if mode == "replacement" and (background_video is None or mask_video is None):
            raise ValueError(
                "Provide `background_video` and `mask_video`. Cannot leave both `background_video` and `mask_video` undefined when mode is `replacement`."
            )
        if mode == "replacement" and (not isinstance(background_video, list) or not isinstance(mask_video, list)):
            raise ValueError(
                "`background_video` and `mask_video` must be lists of PIL images when mode is `replacement`."
            )

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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

        if mode is not None and (not isinstance(mode, str) or mode not in ("animation", "replacement")):
            raise ValueError(
                f"`mode` has to be of type `str` and in ('animation', 'replacement') but its type is {type(mode)} and value is {mode}"
            )

        if num_frames_for_temporal_guidance is not None and (
            not isinstance(num_frames_for_temporal_guidance, int) or num_frames_for_temporal_guidance not in (1, 5)
        ):
            raise ValueError(
                f"`num_frames_for_temporal_guidance` has to be of type `int` and 1 or 5 but its type is {type(num_frames_for_temporal_guidance)} and value is {num_frames_for_temporal_guidance}"
            )

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 80,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        conditioning_pixel_values: Optional[torch.Tensor] = None,
        refer_t_pixel_values: Optional[torch.Tensor] = None,
        background_pixel_values: Optional[torch.Tensor] = None,
        mask_pixel_values: Optional[torch.Tensor] = None,
        mask_reft_len: Optional[int] = None,
        mode: Optional[str] = None,
        y_ref: Optional[str] = None,
        calculate_noise_latents_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_latent_frames = num_frames // self.vae_scale_factor_temporal + 1
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

        # Prepare latent normalization parameters
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        # The first outer loop
        if mask_reft_len == 0:
            image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]
            image = image.to(device=device, dtype=self.vae.dtype)
            # Encode conditioning (pose) video
            conditioning_pixel_values = conditioning_pixel_values.to(device=device, dtype=self.vae.dtype)

            if isinstance(generator, list):
                ref_latents = [retrieve_latents(self.vae.encode(image), sample_mode="argmax") for _ in generator]
                ref_latents = torch.cat(ref_latents)
                pose_latents = [
                    retrieve_latents(self.vae.encode(conditioning_pixel_values), sample_mode="argmax")
                    for _ in generator
                ]
                pose_latents = torch.cat(pose_latents)
            else:
                ref_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
                ref_latents = ref_latents.repeat(batch_size, 1, 1, 1, 1)
                pose_latents = retrieve_latents(self.vae.encode(conditioning_pixel_values), sample_mode="argmax")
                pose_latents = pose_latents.repeat(batch_size, 1, 1, 1, 1)

            ref_latents = (ref_latents.to(dtype) - latents_mean) * latents_std
            pose_latents = (pose_latents.to(dtype) - latents_mean) * latents_std

            mask_ref = self.get_i2v_mask(batch_size, 1, latent_height, latent_width, 1, None, device)
            y_ref = torch.concat([mask_ref, ref_latents], dim=1)

        refer_t_pixel_values = refer_t_pixel_values.to(self.vae.dtype)
        background_pixel_values = background_pixel_values.to(self.vae.dtype)

        if mode == "replacement" and mask_pixel_values is not None:
            mask_pixel_values = 1 - mask_pixel_values
            mask_pixel_values = mask_pixel_values.flatten(0, 1)
            mask_pixel_values = F.interpolate(mask_pixel_values, size=(latent_height, latent_width), mode="nearest")
            mask_pixel_values = mask_pixel_values.unflatten(0, (-1, 1))

        if mask_reft_len > 0 and not calculate_noise_latents_only:
            if mode == "replacement":
                y_reft = retrieve_latents(
                    self.vae.encode(
                        torch.concat(
                            [
                                refer_t_pixel_values[:, :, :mask_reft_len],
                                background_pixel_values[:, :, mask_reft_len:],
                            ],
                            dim=2,
                        )
                    ),
                    sample_mode="argmax",
                )
            else:
                y_reft = retrieve_latents(
                    self.vae.encode(
                        torch.concat(
                            [
                                F.interpolate(
                                    refer_t_pixel_values[:, :, :mask_reft_len], size=(height, width), mode="bicubic"
                                ),
                                torch.zeros(
                                    batch_size,
                                    3,
                                    num_frames - mask_reft_len,
                                    height,
                                    width,
                                    device=device,
                                    dtype=self.vae.dtype,
                                ),
                            ],
                            dim=2,
                        )
                    ),
                    sample_mode="argmax",
                )
        elif mask_reft_len == 0 and not calculate_noise_latents_only:
            if mode == "replacement":
                y_reft = retrieve_latents(self.vae.encode(background_pixel_values), sample_mode="argmax")
            else:
                y_reft = retrieve_latents(
                    self.vae.encode(
                        torch.zeros(
                            batch_size,
                            3,
                            num_frames - mask_reft_len,
                            height,
                            width,
                            device=device,
                            dtype=self.vae.dtype,
                        )
                    ),
                    sample_mode="argmax",
                )

        if mask_reft_len == 0 or not calculate_noise_latents_only:
            y_reft = (y_reft.to(dtype) - latents_mean) * latents_std
            msk_reft = self.get_i2v_mask(
                batch_size, num_latent_frames, latent_height, latent_width, mask_reft_len, mask_pixel_values, device
            )

            y_reft = torch.concat([msk_reft, y_reft], dim=1)
            condition = torch.concat([y_ref, y_reft], dim=2)

        if mask_reft_len == 0 and not calculate_noise_latents_only:
            return latents, condition, pose_latents, y_ref, mask_pixel_values
        elif mask_reft_len > 0 and not calculate_noise_latents_only:
            return latents, condition
        elif mask_reft_len > 0 and calculate_noise_latents_only:
            return latents

    def get_i2v_mask(
        self, batch_size, latent_t, latent_h, latent_w, mask_len=1, mask_pixel_values=None, device="cuda"
    ):
        if mask_pixel_values is None:
            mask_lat_size = torch.zeros(batch_size, 1, (latent_t - 1) * 4 + 1, latent_h, latent_w, device=device)
        else:
            mask_lat_size = mask_pixel_values.clone()
        mask_lat_size[:, :, :mask_len] = 1
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, self.vae_scale_factor_temporal, latent_h, latent_w
        ).transpose(1, 2)

        return mask_lat_size

    def pad_video(self, frames, num_target_frames):
        """
        pad_video([1, 2, 3, 4, 5], 10) -> [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
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
        height: int = 480,
        width: int = 832,
        num_frames: int = 76,
        num_inference_steps: int = 20,
        mode: str = "animation",
        num_frames_for_temporal_guidance: int = 1,
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
                When mode is `"replacement"`, the input background video to condition the generation on. Must be a list
                of PIL images.
            mask_video (`List[PIL.Image.Image]`, *optional*):
                When mode is `"replacement"`, the input mask video to condition the generation on. Must be a list of
                PIL images.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            mode (`str`, defaults to `"animation"`):
                The mode of the generation. Choose between `"animation"` and `"replacement"`.
            num_frames_for_temporal_guidance (`int`, defaults to `1`):
                The number of frames used for temporal guidance. Recommended to be 1 or 5.
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `80`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
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
            num_frames_for_temporal_guidance,
        )

        if num_frames % self.vae_scale_factor_temporal != 0:
            logger.warning(
                f"`num_frames` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal
        num_frames = max(num_frames, 1)

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

        # Encode image embedding
        if image_embeds is None:
            image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # Calculate the number of valid frames
        num_real_frames = len(pose_video)
        real_clip_len = num_frames - num_frames_for_temporal_guidance
        last_clip_num = (num_real_frames - num_frames_for_temporal_guidance) % real_clip_len
        if last_clip_num == 0:
            extra = 0
        else:
            extra = real_clip_len - last_clip_num
        num_target_frames = num_real_frames + extra

        pose_video = self.pad_video(pose_video, num_target_frames)
        face_video = self.pad_video(face_video, num_target_frames)
        if mode == "replacement":
            background_video = self.pad_video(background_video, num_target_frames)
            mask_video = self.pad_video(mask_video, num_target_frames)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        # Get dimensions from the first frame of pose_video (PIL Image.size returns (width, height))
        width, height = pose_video[0].size
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)

        pose_video = self.video_processor.preprocess_video(pose_video, height=height, width=width).to(
            device, dtype=torch.float32
        )
        face_video = self.video_processor.preprocess_video(face_video, height=height, width=width).to(
            device, dtype=torch.float32
        )
        if mode == "replacement":
            background_video = self.video_processor.preprocess_video(background_video, height=height, width=width).to(
                device, dtype=torch.float32
            )
            mask_video = self.video_processor_for_mask.preprocess_video(mask_video, height=height, width=width).to(
                device, dtype=torch.float32
            )

        start = 0
        end = num_frames
        all_out_frames = []
        out_frames = None
        y_ref = None
        calculate_noise_latents_only = False

        while True:
            if start + num_frames_for_temporal_guidance >= len(pose_video):
                break

            if start == 0:
                mask_reft_len = 0
            else:
                mask_reft_len = num_frames_for_temporal_guidance

            conditioning_pixel_values = pose_video[start:end]
            face_pixel_values = face_video[start:end]

            if start == 0:
                refer_t_pixel_values = torch.zeros(image.shape[0], 3, num_frames_for_temporal_guidance, height, width)
            elif start > 0:
                refer_t_pixel_values = (
                    out_frames[0, :, -num_frames_for_temporal_guidance:].clone().detach().unsqueeze(0)
                )

            if mode == "replacement":
                background_pixel_values = background_video[start:end]
                mask_pixel_values = mask_video[start:end].permute(0, 2, 1, 3, 4)
            else:
                mask_pixel_values = None
                background_pixel_values = None

            latents_outputs = self.prepare_latents(
                image if start == 0 else None,
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents if start == 0 else None,
                conditioning_pixel_values,
                refer_t_pixel_values,
                background_pixel_values,
                mask_pixel_values if not calculate_noise_latents_only else None,
                mask_reft_len,
                mode,
                y_ref if start > 0 and not calculate_noise_latents_only else None,
                calculate_noise_latents_only,
            )
            # First iteration
            if start == 0:
                latents, condition, pose_latents, y_ref, mask_pixel_values = latents_outputs
            # Second iteration
            elif start > 0 and not calculate_noise_latents_only:
                latents, condition = latents_outputs
                calculate_noise_latents_only = True
            # Subsequent iterations
            else:
                latents = latents_outputs

            # 6. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            pose_hidden_states=pose_latents,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            face_pixel_values=face_pixel_values,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    if self.do_classifier_free_guidance:
                        # Blank out face for unconditional guidance (set all pixels to -1)
                        face_pixel_values_uncond = face_pixel_values * 0 - 1
                        with self.transformer.cache_context("uncond"):
                            noise_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                pose_hidden_states=pose_latents,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                face_pixel_values=face_pixel_values_uncond,
                                encoder_hidden_states_image=image_embeds,
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
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            # Skip the first latent frame (used for conditioning)
            out_frames = self.vae.decode(latents[:, :, 1:], return_dict=False)[0]

            if start > 0:
                out_frames = out_frames[:, :, num_frames_for_temporal_guidance:]
            all_out_frames.append(out_frames)

            start += num_frames - num_frames_for_temporal_guidance
            end += num_frames - num_frames_for_temporal_guidance

        self._current_timestep = None

        if not output_type == "latent":
            video = torch.cat(all_out_frames, dim=2)[:, :, :num_real_frames]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            # TODO
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
