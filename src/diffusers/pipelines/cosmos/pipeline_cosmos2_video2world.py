# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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
from transformers import T5EncoderModel, T5TokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLWan, CosmosTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_cosmos_guardrail_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import CosmosPipelineOutput


if is_cosmos_guardrail_available():
    from cosmos_guardrail import CosmosSafetyChecker
else:

    class CosmosSafetyChecker:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "`cosmos_guardrail` is not installed. Please install it to use the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )


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
        >>> from diffusers import Cosmos2VideoToWorldPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> # Available checkpoints: nvidia/Cosmos-Predict2-2B-Video2World, nvidia/Cosmos-Predict2-14B-Video2World
        >>> model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
        >>> pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
        >>> negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yellow-scrubber.png"
        ... )

        >>> video = pipe(
        ...     image=image, prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
        ... ).frames[0]
        >>> export_to_video(video, "output.mp4", fps=16)
        ```
"""


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


class Cosmos2VideoToWorldPipeline(DiffusionPipeline):
    r"""
    Pipeline for video-to-world generation using [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. Cosmos uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-11b](https://huggingface.co/google-t5/t5-11b) variant.
        tokenizer (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CosmosTransformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # We mark safety_checker as optional here to get around some test failures, but it is not really optional
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: CosmosTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        safety_checker: CosmosSafetyChecker = None,
    ):
        super().__init__()

        if safety_checker is None:
            safety_checker = CosmosSafetyChecker()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.sigma_max = 80.0
        self.sigma_min = 0.002
        self.sigma_data = 1.0
        self.final_sigmas_type = "sigma_min"
        if self.scheduler is not None:
            self.scheduler.register_to_config(
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                sigma_data=self.sigma_data,
                final_sigmas_type=self.final_sigmas_type,
            )

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_length=True,
            return_offsets_mapping=False,
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=prompt_attention_mask
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        lengths = prompt_attention_mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            prompt_embeds[i, length:] = 0

        return prompt_embeds

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
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
                prompt=prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

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
                prompt=negative_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        video: torch.Tensor,
        batch_size: int,
        num_channels_latents: 16,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        do_classifier_free_guidance: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_cond_frames = video.size(2)
        if num_cond_frames >= num_frames:
            # Take the last `num_frames` frames for conditioning
            num_cond_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            video = video[:, :, -num_frames:]
        else:
            num_cond_latent_frames = (num_cond_frames - 1) // self.vae_scale_factor_temporal + 1
            num_padding_frames = num_frames - num_cond_frames
            last_frame = video[:, :, -1:]
            padding = last_frame.repeat(1, 1, num_padding_frames, 1, 1)
            video = torch.cat([video, padding], dim=2)

        if isinstance(generator, list):
            init_latents = [
                retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator=generator[i])
                for i in range(batch_size)
            ]
        else:
            init_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        init_latents = (init_latents - latents_mean) / latents_std * self.scheduler.config.sigma_data

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.config.sigma_max

        padding_shape = (batch_size, 1, num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

        uncond_indicator = uncond_mask = None
        if do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding

        return latents, init_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
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

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

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
        image: PipelineImageInput = None,
        video: List[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        fps: int = 16,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        sigma_conditioning: float = 0.0001,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*):
                The image to be used as a conditioning input for the video generation.
            video (`List[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*):
                The video to be used as a conditioning input for the video generation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `704`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `93`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `35`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `7.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`.
            fps (`int`, defaults to `16`):
                The frames per second of the generated video.
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
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`CosmosPipelineOutput`] instead of a plain tuple.
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
                The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
                the prompt is shorter than this length, it will be padded.
            sigma_conditioning (`float`, defaults to `0.0001`):
                The sigma value used for scaling conditioning latents. Ideally, it should not be changed or should be
                set to a small value close to zero.

        Examples:

        Returns:
            [`~CosmosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`CosmosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if self.safety_checker is None:
            raise ValueError(
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if self.safety_checker is not None:
            self.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )
            self.safety_checker.to("cpu")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare timesteps
        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
        if self.scheduler.config.final_sigmas_type == "sigma_min":
            # Replace the last sigma (which is zero) with the minimum sigma value
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        # 5. Prepare latent variables
        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        if image is not None:
            video = self.video_processor.preprocess(image, height, width).unsqueeze(2)
        else:
            video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=vae_dtype)

        num_channels_latents = self.transformer.config.in_channels - 1
        latents, conditioning_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask = self.prepare_latents(
            video,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            self.do_classifier_free_guidance,
            torch.float32,
            device,
            generator,
            latents,
        )
        unconditioning_latents = None

        cond_mask = cond_mask.to(transformer_dtype)
        if self.do_classifier_free_guidance:
            uncond_mask = uncond_mask.to(transformer_dtype)
            unconditioning_latents = conditioning_latents

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        sigma_conditioning = torch.tensor(sigma_conditioning, dtype=torch.float32, device=device)
        t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]

                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2), -1, -1
                )  # [B, 1, T, 1, 1]

                cond_latent = latents * c_in
                cond_latent = cond_indicator * conditioning_latents + (1 - cond_indicator) * cond_latent
                cond_latent = cond_latent.to(transformer_dtype)
                cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                cond_timestep = cond_timestep.to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=cond_latent,
                    timestep=cond_timestep,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps,
                    condition_mask=cond_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(transformer_dtype)
                noise_pred = cond_indicator * conditioning_latents + (1 - cond_indicator) * noise_pred

                if self.do_classifier_free_guidance:
                    uncond_latent = latents * c_in
                    uncond_latent = uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * uncond_latent
                    uncond_latent = uncond_latent.to(transformer_dtype)
                    uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                    uncond_timestep = uncond_timestep.to(transformer_dtype)

                    noise_pred_uncond = self.transformer(
                        hidden_states=uncond_latent,
                        timestep=uncond_timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        fps=fps,
                        condition_mask=uncond_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(transformer_dtype)
                    noise_pred_uncond = (
                        uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                    )
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                noise_pred = (latents - noise_pred) / current_sigma
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

        self._current_timestep = None

        if not output_type == "latent":
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

            if self.safety_checker is not None:
                self.safety_checker.to(device)
                video = self.video_processor.postprocess_video(video, output_type="np")
                video = (video * 255).astype(np.uint8)
                video_batch = []
                for vid in video:
                    vid = self.safety_checker.check_video_safety(vid)
                    video_batch.append(vid)
                video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
                video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
                video = self.video_processor.postprocess_video(video, output_type=output_type)
                self.safety_checker.to("cpu")
            else:
                video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)
