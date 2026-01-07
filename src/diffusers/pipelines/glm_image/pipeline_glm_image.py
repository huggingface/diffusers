# Copyright 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
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
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import AutoTokenizer, T5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...loaders import CogView4LoraLoaderMixin
from ...models import AutoencoderKL, GlmImageTransformer2DModel
from ...models.transformers.transformer_glm_image import GlmImageAttenProcessorState
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_output import GlmImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import GlmImagePipeline

        >>> pipe = GlmImagePipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        >>> image.save("output.png")
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    base_shift: float = 0.25,
    max_shift: float = 0.75,
) -> float:
    m = (image_seq_len / base_seq_len) ** 0.5
    mu = m * max_shift + base_shift
    return mu


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
    accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())

    if timesteps is not None and sigmas is not None:
        if not accepts_timesteps and not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep or sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is not None and sigmas is None:
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is None and sigmas is not None:
        if not accepts_sigmas:
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


class GlmImagePipeline(DiffusionPipeline, CogView4LoraLoaderMixin):
    r"""
    Pipeline for text-to-image generation using CogView4.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`GlmModel`]):
            Frozen text-encoder. CogView4 uses [GLM-Image](https://huggingface.co/zai-org/GLM-Image).
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer of class
            [PreTrainedTokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).
        transformer ([`CogView4Transformer2DModel`]):
            A text conditioned `CogView4Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: GlmImageTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer")
            and self.transformer is not None
            and hasattr(self.transformer.config, "sample_size")
            else 128
        )

    def get_glyph_texts(
        self,
        prompt,
    ):
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        ocr_texts = (
            re.findall(r"'([^']*)'", prompt)
            + re.findall(r"“([^“”]*)”", prompt)
            + re.findall(r'"([^"]*)"', prompt)
            + re.findall(r"「([^「」]*)」", prompt)
        )
        return ocr_texts

    def _get_glyph_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        glyph_texts = self.get_glyph_texts(prompt)
        input_ids = self.tokenizer(
            glyph_texts if len(glyph_texts) > 0 else [""],
            max_length=max_sequence_length,
            truncation=True,
        ).input_ids
        input_ids = [
            [self.tokenizer.pad_token_id] * ((len(input_ids) + 1) % 2) + input_ids_ for input_ids_ in input_ids
        ]
        max_length = max(len(input_ids_) for input_ids_ in input_ids)
        attention_mask = torch.tensor(
            [[1] * len(input_ids_) + [0] * (max_length - len(input_ids_)) for input_ids_ in input_ids], device=device
        )
        input_ids = torch.tensor(
            [input_ids_ + [self.tokenizer.pad_token_id] * (max_length - len(input_ids_)) for input_ids_ in input_ids],
            device=device,
        )
        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)

        return glyph_embeds.to(device=device, dtype=dtype)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 2048,
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
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images that should be generated per prompt. torch device to place the resulting embeddings on
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
            max_sequence_length (`int`, defaults to `2048`):
                Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_glyph_embeds(prompt, max_sequence_length, device, dtype)

        seq_len = prompt_embeds.size(1)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

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

            negative_prompt_embeds = self._get_glyph_embeds(negative_prompt, max_sequence_length, device, dtype)

            seq_len = negative_prompt_embeds.size(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if latents is not None:
            return latents.to(device)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * self.transformer.config.patch_size) != 0
            or width is not None
            and width % (self.transformer.config.patch_size) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

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

    @property
    def guidance_scale(self):
        return self._guidance_scale

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
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prior_token_id: Optional[torch.LongTensor] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        condition_images: Optional[
            Union[
                torch.Tensor, PIL.Image.Image, np.ndarray, List[torch.Tensor], List[PIL.Image.Image], List[np.ndarray]
            ]
        ] = None,
        condition_images_prior_token_id: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 2048,
    ) -> Union[GlmImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. If not provided, it is set to 2048.
            width (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. If not provided it is set to 2048.
            num_inference_steps (`int`, *optional*, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to `1`):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            original_size (`Tuple[int]`, *optional*, defaults to (2048, 2048)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `224`):
                Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.

        Examples:

        Returns:
            [`~pipelines.cogview4.pipeline_CogView4.CogView4PipelineOutput`] or `tuple`:
            [`~pipelines.cogview4.pipeline_CogView4.CogView4PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        assert batch_size == 1, "batch_size must be 1"

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=self.dtype,
        )

        # 4. process images
        if condition_images is not None and not isinstance(condition_images, list):
            condition_images = [condition_images]
            condition_images_prior_token_id = [condition_images_prior_token_id]
        assert condition_images is None or (len(condition_images) == len(condition_images_prior_token_id)), (
            "image and image_prior_token_id must be the same length"
        )

        if condition_images is not None:
            preprocessed_condition_images = []
            for img in condition_images:
                image_height, image_width = img.size[::-1] if isinstance(img, PIL.Image.Image) else img.shape[:2]
                multiple_of = self.vae_scale_factor * self.transformer.config.patch_size
                image_height = (image_height // multiple_of) * multiple_of
                image_width = (image_width // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width)
                preprocessed_condition_images.append(img)
                height = height or image_height
                width = width or image_width
            condition_images = preprocessed_condition_images
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. Prepare latents and (optional) condition_images kv cache
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        if condition_images is not None:
            self.transformer.set_attention_processors_state(GlmImageAttenProcessorState.ImageEditWriteKV)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.latent_channels, 1, 1)
                .to(self.vae.device, self.vae.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.latent_channels, 1, 1)
                .to(self.vae.device, self.vae.dtype)
            )
            empty_glyph_hiddens = torch.zeros_like(prompt_embeds)[:1, :0, ...]
            for condition_image, condition_image_prior_token_id in zip(
                condition_images, condition_images_prior_token_id
            ):
                condition_image = condition_image.to(device=device, dtype=self.vae.dtype)
                condition_latent = retrieve_latents(
                    self.vae.encode(condition_image), generator=generator, sample_mode="argmax"
                )
                condition_latent = (condition_latent - latents_mean) / latents_std
                _ = self.transformer(
                    hidden_states=condition_latent,
                    glyph_hidden_states=empty_glyph_hiddens,
                    prior_token_id=condition_image_prior_token_id,
                    prior_token_drop=torch.full_like(condition_image_prior_token_id, False, dtype=torch.bool),
                    timestep=torch.zeros((1,), device=device),
                    original_size=torch.tensor([condition_image.shape[-2:]], device=device),
                    target_size=torch.tensor([condition_image.shape[-2:]], device=device),
                    crop_coords=torch.zeros((1, 2), device=device),
                    attention_kwargs=attention_kwargs,
                )

        # 6. Prepare additional timestep conditions
        original_size = original_size or (height, width)
        target_size = (height, width)
        original_size = torch.tensor([original_size], dtype=prompt_embeds.dtype, device=device)
        target_size = torch.tensor([target_size], dtype=prompt_embeds.dtype, device=device)
        crops_coords_top_left = torch.tensor([crops_coords_top_left], dtype=prompt_embeds.dtype, device=device)

        original_size = original_size.repeat(batch_size * num_images_per_prompt, 1)
        target_size = target_size.repeat(batch_size * num_images_per_prompt, 1)
        crops_coords_top_left = crops_coords_top_left.repeat(batch_size * num_images_per_prompt, 1)

        # Prepare timesteps
        image_seq_len = ((height // self.vae_scale_factor) * (width // self.vae_scale_factor)) // (
            self.transformer.config.patch_size**2
        )
        timesteps = (
            np.linspace(self.scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1)[:-1]
            if timesteps is None
            else np.array(timesteps)
        )
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps if sigmas is None else sigmas
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("base_shift", 0.25),
            self.scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # 7. Denoising loop
        transformer_dtype = self.transformer.dtype
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        prior_token_drop_cond = torch.full_like(prior_token_id, False, dtype=torch.bool)
        prior_token_drop_uncond = torch.full_like(prior_token_id, True, dtype=torch.bool)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]) - 1

                if condition_images is not None:
                    self.transformer.set_attention_processors_state(GlmImageAttenProcessorState.ImageEditReadKV)

                noise_pred_cond = self.transformer(
                    hidden_states=latent_model_input,
                    glyph_hidden_states=prompt_embeds,
                    prior_token_id=prior_token_id,
                    prior_token_drop=prior_token_drop_cond,
                    timestep=timestep,
                    original_size=original_size,
                    target_size=target_size,
                    crop_coords=crops_coords_top_left,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0].float()

                # perform guidance
                if self.do_classifier_free_guidance:
                    if condition_images is not None:
                        self.transformer.set_attention_processors_state(
                            GlmImageAttenProcessorState.ImageEditDontReadKV
                        )
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        glyph_hidden_states=negative_prompt_embeds,
                        prior_token_id=prior_token_id,
                        prior_token_drop=prior_token_drop_uncond,
                        timestep=timestep,
                        original_size=original_size,
                        target_size=target_size,
                        crop_coords=crops_coords_top_left,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0].float()

                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, self.scheduler.sigmas[i], callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        self.transformer.set_attention_processors_state(GlmImageAttenProcessorState.ImageGen)
        self.transformer.clear_attention_processors_cache()

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.latent_channels, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.latent_channels, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std + latents_mean
            condition_images = self.vae.decode(latents, return_dict=False, generator=generator)[0]
        else:
            condition_images = latents

        condition_images = self.image_processor.postprocess(condition_images, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (condition_images,)

        return GlmImagePipelineOutput(images=condition_images)
