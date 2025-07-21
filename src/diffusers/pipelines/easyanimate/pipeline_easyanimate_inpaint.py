# Copyright 2025 The EasyAnimate team and The HuggingFace Team.
# All rights reserved.
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
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    BertModel,
    BertTokenizer,
    Qwen2Tokenizer,
    Qwen2VLForConditionalGeneration,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLMagvit, EasyAnimateTransformer3DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from .pipeline_output import EasyAnimatePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import EasyAnimateInpaintPipeline
        >>> from diffusers.pipelines.easyanimate.pipeline_easyanimate_inpaint import get_image_to_video_latent
        >>> from diffusers.utils import export_to_video, load_image

        >>> pipe = EasyAnimateInpaintPipeline.from_pretrained(
        ...     "alibaba-pai/EasyAnimateV5.1-12b-zh-InP-diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        >>> validation_image_start = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )

        >>> validation_image_end = None
        >>> sample_size = (448, 576)
        >>> num_frames = 49
        >>> input_video, input_video_mask = get_image_to_video_latent(
        ...     [validation_image_start], validation_image_end, num_frames, sample_size
        ... )

        >>> video = pipe(
        ...     prompt,
        ...     num_frames=num_frames,
        ...     negative_prompt="Twisted body, limb deformities, text subtitles, comics, stillness, ugliness, errors, garbled text.",
        ...     height=sample_size[0],
        ...     width=sample_size[1],
        ...     video=input_video,
        ...     mask_video=input_video_mask,
        ... )
        >>> export_to_video(video.frames[0], "output.mp4", fps=8)
        ```
"""


def preprocess_image(image, sample_size):
    """
    Preprocess a single image (PIL.Image, numpy.ndarray, or torch.Tensor) to a resized tensor.
    """
    if isinstance(image, torch.Tensor):
        # If input is a tensor, assume it's in CHW format and resize using interpolation
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=sample_size, mode="bilinear", align_corners=False
        ).squeeze(0)
    elif isinstance(image, Image.Image):
        # If input is a PIL image, resize and convert to numpy array
        image = image.resize((sample_size[1], sample_size[0]))
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # If input is a numpy array, resize using PIL
        image = Image.fromarray(image).resize((sample_size[1], sample_size[0]))
        image = np.array(image)
    else:
        raise ValueError("Unsupported input type. Expected PIL.Image, numpy.ndarray, or torch.Tensor.")

    # Convert to tensor if not already
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC -> CHW, normalize to [0, 1]

    return image


def get_image_to_video_latent(validation_image_start, validation_image_end, num_frames, sample_size):
    """
    Generate latent representations for video from start and end images. Inputs can be PIL.Image, numpy.ndarray, or
    torch.Tensor.
    """
    input_video = None
    input_video_mask = None

    if validation_image_start is not None:
        # Preprocess the starting image(s)
        if isinstance(validation_image_start, list):
            image_start = [preprocess_image(img, sample_size) for img in validation_image_start]
        else:
            image_start = preprocess_image(validation_image_start, sample_size)

        # Create video tensor from the starting image(s)
        if isinstance(image_start, list):
            start_video = torch.cat(
                [img.unsqueeze(1).unsqueeze(0) for img in image_start],
                dim=2,
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, num_frames, 1, 1])
            input_video[:, :, : len(image_start)] = start_video
        else:
            input_video = torch.tile(
                image_start.unsqueeze(1).unsqueeze(0),
                [1, 1, num_frames, 1, 1],
            )

        # Normalize input video (already normalized in preprocess_image)

        # Create mask for the input video
        input_video_mask = torch.zeros_like(input_video[:, :1])
        if isinstance(image_start, list):
            input_video_mask[:, :, len(image_start) :] = 255
        else:
            input_video_mask[:, :, 1:] = 255

        # Handle ending image(s) if provided
        if validation_image_end is not None:
            if isinstance(validation_image_end, list):
                image_end = [preprocess_image(img, sample_size) for img in validation_image_end]
                end_video = torch.cat(
                    [img.unsqueeze(1).unsqueeze(0) for img in image_end],
                    dim=2,
                )
                input_video[:, :, -len(end_video) :] = end_video
                input_video_mask[:, :, -len(image_end) :] = 0
            else:
                image_end = preprocess_image(validation_image_end, sample_size)
                input_video[:, :, -1:] = image_end.unsqueeze(1).unsqueeze(0)
                input_video_mask[:, :, -1:] = 0

    elif validation_image_start is None:
        # If no starting image is provided, initialize empty tensors
        input_video = torch.zeros([1, 3, num_frames, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, num_frames, sample_size[0], sample_size[1]]) * 255

    return input_video, input_video_mask


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Resize mask information in magvit
def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :], size=target_size, mode="trilinear", align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :], size=target_size, mode="trilinear", align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(mask, size=target_size, mode="trilinear", align_corners=False)
    return resized_mask


## Add noise to reference video
def add_noise_to_reference_video(image, ratio=None, generator=None):
    if ratio is None:
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
        sigma = torch.exp(sigma).to(image.dtype)
    else:
        sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio

    if generator is not None:
        image_noise = (
            torch.randn(image.size(), generator=generator, dtype=image.dtype, device=image.device)
            * sigma[:, None, None, None, None]
        )
    else:
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
    image_noise = torch.where(image == -1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class EasyAnimateInpaintPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using EasyAnimate.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    EasyAnimate uses one text encoder [qwen2 vl](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) in V5.1.

    Args:
        vae ([`AutoencoderKLMagvit`]):
            Variational Auto-Encoder (VAE) Model to encode and decode video to and from latent representations.
        text_encoder (Optional[`~transformers.Qwen2VLForConditionalGeneration`, `~transformers.BertModel`]):
            EasyAnimate uses [qwen2 vl](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) in V5.1.
        tokenizer (Optional[`~transformers.Qwen2Tokenizer`, `~transformers.BertTokenizer`]):
            A `Qwen2Tokenizer` or `BertTokenizer` to tokenize text.
        transformer ([`EasyAnimateTransformer3DModel`]):
            The EasyAnimate model designed by EasyAnimate Team.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with EasyAnimate to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLMagvit,
        text_encoder: Union[Qwen2VLForConditionalGeneration, BertModel],
        tokenizer: Union[Qwen2Tokenizer, BertTokenizer],
        transformer: EasyAnimateTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.enable_text_attention_mask = (
            self.transformer.config.enable_text_attention_mask
            if getattr(self, "transformer", None) is not None
            else True
        )
        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 4
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    # Copied from diffusers.pipelines.easyanimate.pipeline_easyanimate.EasyAnimatePipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            dtype (`torch.dtype`):
                torch dtype
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
        """
        dtype = dtype or self.text_encoder.dtype
        device = device or self.text_encoder.device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(prompt, str):
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": _prompt}],
                    }
                    for _prompt in prompt
                ]
            text = [
                self.tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages
            ]

            text_inputs = self.tokenizer(
                text=text,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                padding_side="right",
                return_tensors="pt",
            )
            text_inputs = text_inputs.to(self.text_encoder.device)

            text_input_ids = text_inputs.input_ids
            prompt_attention_mask = text_inputs.attention_mask
            if self.enable_text_attention_mask:
                # Inference: Generation of the output
                prompt_embeds = self.text_encoder(
                    input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
                ).hidden_states[-2]
            else:
                raise ValueError("LLM needs attention_mask")
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.to(device=device)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is not None and isinstance(negative_prompt, str):
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": negative_prompt}],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": _negative_prompt}],
                    }
                    for _negative_prompt in negative_prompt
                ]
            text = [
                self.tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages
            ]

            text_inputs = self.tokenizer(
                text=text,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                padding_side="right",
                return_tensors="pt",
            )
            text_inputs = text_inputs.to(self.text_encoder.device)

            text_input_ids = text_inputs.input_ids
            negative_prompt_attention_mask = text_inputs.attention_mask
            if self.enable_text_attention_mask:
                # Inference: Generation of the output
                negative_prompt_embeds = self.text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=negative_prompt_attention_mask,
                    output_hidden_states=True,
                ).hidden_states[-2]
            else:
                raise ValueError("LLM needs attention_mask")
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device=device)

        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        callback_on_step_end_tensor_inputs=None,
    ):
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
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
        noise_aug_strength,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        if mask is not None:
            mask = mask.to(device=device, dtype=dtype)
            new_mask = []
            bs = 1
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = self.vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim=0)
            mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=dtype)
            if self.transformer.config.add_noise_in_inpaint_model:
                masked_image = add_noise_to_reference_video(
                    masked_image, ratio=noise_aug_strength, generator=generator
                )
            new_mask_pixel_values = []
            bs = 1
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim=0)
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        latents=None,
        video=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_video_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_temporal_compression_ratio + 1,
            height // self.vae_spatial_compression_ratio,
            width // self.vae_spatial_compression_ratio,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if return_video_latents or (latents is None and not is_strength_max):
            video = video.to(device=device, dtype=dtype)
            bs = 1
            new_video = []
            for i in range(0, video.shape[0], bs):
                video_bs = video[i : i + bs]
                video_bs = self.vae.encode(video_bs)[0]
                video_bs = video_bs.sample()
                new_video.append(video_bs)
            video = torch.cat(new_video, dim=0)
            video = video * self.vae.config.scaling_factor

            video_latents = video.repeat(batch_size // video.shape[0], 1, 1, 1, 1)
            video_latents = video_latents.to(device=device, dtype=dtype)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                latents = noise if is_strength_max else self.scheduler.scale_noise(video_latents, timestep, noise)
            else:
                latents = noise if is_strength_max else self.scheduler.add_noise(video_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            if hasattr(self.scheduler, "init_noise_sigma"):
                noise = latents.to(device)
                latents = noise * self.scheduler.init_noise_sigma
            else:
                latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_video_latents:
            outputs += (video_latents,)

        return outputs

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = 49,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        masked_video_latents: Union[torch.FloatTensor] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        strength: float = 1.0,
        noise_aug_strength: float = 0.0563,
        timesteps: Optional[List[int]] = None,
    ):
        r"""
        The call function to the pipeline for generation with HunyuanDiT.

        Examples:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            num_frames (`int`, *optional*):
                Length of the video to be generated in seconds. This parameter influences the number of frames and
                continuity of generated content.
            video (`torch.FloatTensor`, *optional*):
                A tensor representing an input video, which can be modified depending on the prompts provided.
            mask_video (`torch.FloatTensor`, *optional*):
                A tensor to specify areas of the video to be masked (omitted from generation).
            masked_video_latents (`torch.FloatTensor`, *optional*):
                Latents from masked portions of the video, utilized during image generation.
            height (`int`, *optional*):
                The height in pixels of the generated image or video frames.
            width (`int`, *optional*):
                The width in pixels of the generated image or video frames.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image but slower
                inference time. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is effective when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to exclude in image generation. If not defined, you need to provide
                `negative_prompt_embeds`. This parameter is ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                A parameter defined in the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies to the
                [`~schedulers.DDIMScheduler`] and is ignored in other schedulers. It adjusts noise level during the
                inference process.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) for setting
                random seeds which helps in making generation deterministic.
            latents (`torch.Tensor`, *optional*):
                A pre-computed latent representation which can be used to guide the generation process.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings, aiding in fine-tuning what should not be represented in the
                outputs. If not provided, embeddings are generated from the `negative_prompt` argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask guiding the focus of the model on specific parts of the prompt text. Required when using
                `prompt_embeds`.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt, needed when `negative_prompt_embeds` are used.
            output_type (`str`, *optional*, defaults to `"latent"`):
                The output format of the generated image. Choose between `PIL.Image` and `np.array` to define how you
                want the results to be formatted.
            return_dict (`bool`, *optional*, defaults to `True`):
                If set to `True`, a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] will be returned;
                otherwise, a tuple containing the generated images and safety flags will be returned.
            callback_on_step_end (`Callable[[int, int, Dict], None]`, `PipelineCallback`, `MultiPipelineCallbacks`,
            *optional*):
                A callback function (or a list of them) that will be executed at the end of each denoising step,
                allowing for custom processing during generation.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Specifies which tensor inputs should be included in the callback function. If not defined, all tensor
                inputs will be passed, facilitating enhanced logging or monitoring of the generation process.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Rescale parameter for adjusting noise configuration based on guidance rescale. Based on findings from
                [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://huggingface.co/papers/2305.08891).
            strength (`float`, *optional*, defaults to 1.0):
                Affects the overall styling or quality of the generated output. Values closer to 1 usually provide
                direct adherence to prompts.

        Examples:
            # Example usage of the function for generating images based on prompts.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                Returns either a structured output containing generated images and their metadata when `return_dict` is
                `True`, or a simpler tuple, where the first element is a list of generated images and the second
                element indicates if any of them contain "not-safe-for-work" (NSFW) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        height = int(height // 16 * 16)
        width = int(width // 16 * 16)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        else:
            dtype = self.transformer.dtype

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 4. set timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, mu=1
            )
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )

        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        if video is not None:
            batch_size, channels, num_frames, height_video, width_video = video.shape
            init_video = self.image_processor.preprocess(
                video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height_video, width_video),
                height=height,
                width=width,
            )
            init_video = init_video.to(dtype=torch.float32)
            init_video = init_video.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        else:
            init_video = None

        # Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer.config.in_channels
        return_image_latents = num_channels_transformer == num_channels_latents

        # 5. Prepare latents.
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            dtype,
            device,
            generator,
            latents,
            video=init_video,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_video_latents=return_image_latents,
        )
        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 6. Prepare inpaint latents if it needs.
        if mask_video is not None:
            if (mask_video == 255).all():
                mask = torch.zeros_like(latents).to(device, dtype)
                # Use zero latents if we want to t2v.
                if self.transformer.config.resize_inpaint_mask_directly:
                    mask_latents = torch.zeros_like(latents)[:, :1].to(device, dtype)
                else:
                    mask_latents = torch.zeros_like(latents).to(device, dtype)
                masked_video_latents = torch.zeros_like(latents).to(device, dtype)

                mask_input = torch.cat([mask_latents] * 2) if self.do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if self.do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(dtype)
            else:
                # Prepare mask latent variables
                batch_size, channels, num_frames, height_video, width_video = mask_video.shape
                mask_condition = self.mask_processor.preprocess(
                    mask_video.permute(0, 2, 1, 3, 4).reshape(
                        batch_size * num_frames, channels, height_video, width_video
                    ),
                    height=height,
                    width=width,
                )
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = mask_condition.reshape(batch_size, num_frames, channels, height, width).permute(
                    0, 2, 1, 3, 4
                )

                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    if masked_video_latents is None:
                        masked_video = (
                            init_video * (mask_condition_tile < 0.5)
                            + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                        )
                    else:
                        masked_video = masked_video_latents

                    if self.transformer.config.resize_inpaint_mask_directly:
                        _, masked_video_latents = self.prepare_mask_latents(
                            None,
                            masked_video,
                            batch_size,
                            height,
                            width,
                            dtype,
                            device,
                            generator,
                            self.do_classifier_free_guidance,
                            noise_aug_strength=noise_aug_strength,
                        )
                        mask_latents = resize_mask(
                            1 - mask_condition, masked_video_latents, self.vae.config.cache_mag_vae
                        )
                        mask_latents = mask_latents.to(device, dtype) * self.vae.config.scaling_factor
                    else:
                        mask_latents, masked_video_latents = self.prepare_mask_latents(
                            mask_condition_tile,
                            masked_video,
                            batch_size,
                            height,
                            width,
                            dtype,
                            device,
                            generator,
                            self.do_classifier_free_guidance,
                            noise_aug_strength=noise_aug_strength,
                        )

                    mask_input = torch.cat([mask_latents] * 2) if self.do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2)
                        if self.do_classifier_free_guidance
                        else masked_video_latents
                    )
                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(dtype)
                else:
                    inpaint_latents = None

                mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode="trilinear", align_corners=True).to(
                    device, dtype
                )
        else:
            if num_channels_transformer != num_channels_latents:
                mask = torch.zeros_like(latents).to(device, dtype)
                if self.transformer.config.resize_inpaint_mask_directly:
                    mask_latents = torch.zeros_like(latents)[:, :1].to(device, dtype)
                else:
                    mask_latents = torch.zeros_like(latents).to(device, dtype)
                masked_video_latents = torch.zeros_like(latents).to(device, dtype)

                mask_input = torch.cat([mask_latents] * 2) if self.do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if self.do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(dtype)
            else:
                mask = torch.zeros_like(init_video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode="trilinear", align_corners=True).to(
                    device, dtype
                )

                inpaint_latents = None

        # Check that sizes of mask, masked image and latents match
        if num_channels_transformer != num_channels_latents:
            num_channels_mask = mask_latents.shape[1]
            num_channels_masked_image = masked_video_latents.shape[1]
            if (
                num_channels_latents + num_channels_mask + num_channels_masked_image
                != self.transformer.config.in_channels
            ):
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.transformer`: {self.transformer.config} expects"
                    f" {self.transformer.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                    " `pipeline.transformer` or your `mask_image` or `image` input."
                )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])

        # To latents.device
        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                t_expand = torch.tensor([t] * latent_model_input.shape[0], device=device).to(
                    dtype=latent_model_input.dtype
                )

                # predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    t_expand,
                    encoder_hidden_states=prompt_embeds,
                    inpaint_latents=inpaint_latents,
                    return_dict=False,
                )[0]
                if noise_pred.size()[1] != self.vae.config.latent_channels:
                    noise_pred, _ = noise_pred.chunk(2, dim=1)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_transformer == num_channels_latents:
                    init_latents_proper = image_latents
                    init_mask = mask
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                            init_latents_proper = self.scheduler.scale_noise(
                                init_latents_proper, torch.tensor([noise_timestep], noise)
                            )
                        else:
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_proper, noise, torch.tensor([noise_timestep])
                            )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            latents = 1 / self.vae.config.scaling_factor * latents
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return EasyAnimatePipelineOutput(frames=video)
