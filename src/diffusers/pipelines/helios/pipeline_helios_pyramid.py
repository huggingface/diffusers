# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
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
import math
from typing import Any, Callable

import regex as re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import HeliosLoraLoaderMixin
from ...models import AutoencoderKLWan, HeliosTransformer3DModel
from ...schedulers import HeliosDMDScheduler, HeliosScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HeliosPipelineOutput


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
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, HeliosPyramidPipeline

        >>> # Available models: BestWishYsh/Helios-Base, BestWishYsh/Helios-Mid, BestWishYsh/Helios-Distilled
        >>> model_id = "BestWishYsh/Helios-Base"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = HeliosPyramidPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=384,
        ...     width=640,
        ...     num_frames=132,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=24)
        ```
"""


def optimized_scale(positive_flat, negative_flat):
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    return st_star


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


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class HeliosPyramidPipeline(DiffusionPipeline, HeliosLoraLoaderMixin):
    r"""
    Pipeline for text-to-video / image-to-video / video-to-video generation using Helios.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`HeliosTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`HeliosScheduler`, `HeliosDMDScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: HeliosScheduler | HeliosDMDScheduler,
        transformer: HeliosTransformer3DModel,
        is_cfg_zero_star: bool = False,
        is_distilled: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.register_to_config(is_cfg_zero_star=is_cfg_zero_star)
        self.register_to_config(is_distilled=is_distilled)
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # Copied from diffusers.pipelines.helios.pipeline_helios.HeliosPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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

        return prompt_embeds, text_inputs.attention_mask.bool()

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `list[str]`, *optional*):
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
            prompt_embeds, _ = self._get_t5_prompt_embeds(
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

            negative_prompt_embeds, _ = self._get_t5_prompt_embeds(
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
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        image=None,
        video=None,
        guidance_scale=None,
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

        if image is not None and video is not None:
            raise ValueError("image and video cannot be provided simultaneously")

        if guidance_scale > 1.0 and self.config.is_distilled:
            logger.warning(f"Guidance scale {guidance_scale} is ignored for step-wise distilled models.")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 384,
        width: int = 640,
        num_frames: int = 33,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_image_latents(
        self,
        image: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        num_latent_frames_per_chunk: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        fake_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        if latents is None:
            image = image.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
            latents = self.vae.encode(image).latent_dist.sample(generator=generator)
            latents = (latents - latents_mean) * latents_std
        if fake_latents is None:
            min_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
            fake_video = image.repeat(1, 1, min_frames, 1, 1).to(device=device, dtype=self.vae.dtype)
            fake_latents_full = self.vae.encode(fake_video).latent_dist.sample(generator=generator)
            fake_latents_full = (fake_latents_full - latents_mean) * latents_std
            fake_latents = fake_latents_full[:, :, -1:, :, :]
        return latents.to(device=device, dtype=dtype), fake_latents.to(device=device, dtype=dtype)

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        num_latent_frames_per_chunk: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        video = video.to(device=device, dtype=self.vae.dtype)
        if latents is None:
            num_frames = video.shape[2]
            min_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
            num_chunks = num_frames // min_frames
            if num_chunks == 0:
                raise ValueError(
                    f"Video must have at least {min_frames} frames "
                    f"(got {num_frames} frames). "
                    f"Required: (num_latent_frames_per_chunk - 1) * {self.vae_scale_factor_temporal} + 1 = ({num_latent_frames_per_chunk} - 1) * {self.vae_scale_factor_temporal} + 1 = {min_frames}"
                )
            total_valid_frames = num_chunks * min_frames
            start_frame = num_frames - total_valid_frames

            first_frame = video[:, :, 0:1, :, :]
            first_frame_latent = self.vae.encode(first_frame).latent_dist.sample(generator=generator)
            first_frame_latent = (first_frame_latent - latents_mean) * latents_std

            latents_chunks = []
            for i in range(num_chunks):
                chunk_start = start_frame + i * min_frames
                chunk_end = chunk_start + min_frames
                video_chunk = video[:, :, chunk_start:chunk_end, :, :]
                chunk_latents = self.vae.encode(video_chunk).latent_dist.sample(generator=generator)
                chunk_latents = (chunk_latents - latents_mean) * latents_std
                latents_chunks.append(chunk_latents)
            latents = torch.cat(latents_chunks, dim=2)
        return first_frame_latent.to(device=device, dtype=dtype), latents.to(device=device, dtype=dtype)

    def sample_block_noise(
        self,
        batch_size,
        channel,
        num_frames,
        height,
        width,
        patch_size: tuple[int, ...] = (1, 2, 2),
        device: torch.device | None = None,
    ):
        gamma = self.scheduler.config.gamma
        _, ph, pw = patch_size
        block_size = ph * pw

        cov = (
            torch.eye(block_size, device=device) * (1 + gamma)
            - torch.ones(block_size, block_size, device=device) * gamma
        )
        cov += torch.eye(block_size, device=device) * 1e-6
        dist = torch.distributions.MultivariateNormal(torch.zeros(block_size, device=device), covariance_matrix=cov)
        block_number = batch_size * channel * num_frames * (height // ph) * (width // pw)

        noise = dist.sample((block_number,))  # [block number, block_size]
        noise = noise.view(batch_size, channel, num_frames, height // ph, width // pw, ph, pw)
        noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)
        return noise

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

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 384,
        width: int = 640,
        num_frames: int = 132,
        sigmas: list[float] = None,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        # ------------ I2V ------------
        image: PipelineImageInput | None = None,
        image_latents: torch.Tensor | None = None,
        fake_image_latents: torch.Tensor | None = None,
        add_noise_to_image_latents: bool = True,
        image_noise_sigma_min: float = 0.111,
        image_noise_sigma_max: float = 0.135,
        # ------------ V2V ------------
        video: PipelineImageInput | None = None,
        video_latents: torch.Tensor | None = None,
        add_noise_to_video_latents: bool = True,
        video_noise_sigma_min: float = 0.111,
        video_noise_sigma_max: float = 0.135,
        # ------------ Stage 1 ------------
        history_sizes: list = [16, 2, 1],
        num_latent_frames_per_chunk: int = 9,
        keep_first_frame: bool = True,
        is_skip_first_chunk: bool = False,
        # ------------ Stage 2 ------------
        pyramid_num_inference_steps_list: list = [10, 10, 10],
        # ------------ CFG Zero ------------
        use_zero_init: bool | None = True,
        zero_steps: int | None = 1,
        # ------------ DMD ------------
        is_amplify_first_chunk: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `384`):
                The height in pixels of the generated image.
            width (`int`, defaults to `640`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `132`):
                The number of frames in the generated video.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HeliosPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~HeliosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HeliosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        history_sizes = sorted(history_sizes, reverse=True)  # From big to small
        pyramid_num_stages = len(pyramid_num_inference_steps_list)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            image,
            video,
            guidance_scale,
        )

        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        vae_dtype = self.vae.dtype

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            device, self.vae.dtype
        )

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

        # 4. Prepare image or video
        if image is not None:
            image = self.video_processor.preprocess(image, height=height, width=width)
            image_latents, fake_image_latents = self.prepare_image_latents(
                image,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=image_latents,
                fake_latents=fake_image_latents,
            )

        if image_latents is not None and add_noise_to_image_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )
            fake_image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (video_noise_sigma_max - video_noise_sigma_min)
                + video_noise_sigma_min
            )
            fake_image_latents = (
                fake_image_noise_sigma * randn_tensor(fake_image_latents.shape, generator=generator, device=device)
                + (1 - fake_image_noise_sigma) * fake_image_latents
            )

        if video is not None:
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            image_latents, video_latents = self.prepare_video_latents(
                video,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=video_latents,
            )

        if video_latents is not None and add_noise_to_video_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )

            noisy_latents_chunks = []
            num_latent_chunks = video_latents.shape[2] // num_latent_frames_per_chunk
            for i in range(num_latent_chunks):
                chunk_start = i * num_latent_frames_per_chunk
                chunk_end = chunk_start + num_latent_frames_per_chunk
                latent_chunk = video_latents[:, :, chunk_start:chunk_end, :, :]

                chunk_frames = latent_chunk.shape[2]
                frame_sigmas = (
                    torch.rand(chunk_frames, device=device, generator=generator)
                    * (video_noise_sigma_max - video_noise_sigma_min)
                    + video_noise_sigma_min
                )
                frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)

                noisy_chunk = (
                    frame_sigmas * randn_tensor(latent_chunk.shape, generator=generator, device=device)
                    + (1 - frame_sigmas) * latent_chunk
                )
                noisy_latents_chunks.append(noisy_chunk)
            video_latents = torch.cat(noisy_latents_chunks, dim=2)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        window_num_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
        num_latent_chunk = max(1, (num_frames + window_num_frames - 1) // window_num_frames)
        num_history_latent_frames = sum(history_sizes)
        history_video = None
        total_generated_latent_frames = 0

        if not keep_first_frame:
            history_sizes[-1] = history_sizes[-1] + 1
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            num_history_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        if fake_image_latents is not None:
            history_latents = torch.cat([history_latents[:, :, :-1, :, :], fake_image_latents], dim=2)
            total_generated_latent_frames += 1
        if video_latents is not None:
            history_frames = history_latents.shape[2]
            video_frames = video_latents.shape[2]
            if video_frames < history_frames:
                keep_frames = history_frames - video_frames
                history_latents = torch.cat([history_latents[:, :, :keep_frames, :, :], video_latents], dim=2)
            else:
                history_latents = video_latents
            total_generated_latent_frames += video_latents.shape[2]

        if keep_first_frame:
            indices = torch.arange(0, sum([1, *history_sizes, num_latent_frames_per_chunk]))
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
            indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
        else:
            indices = torch.arange(0, sum([*history_sizes, num_latent_frames_per_chunk]))
            (
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_short,
                indices_hidden_states,
            ) = indices.split([*history_sizes, num_latent_frames_per_chunk], dim=0)
        indices_hidden_states = indices_hidden_states.unsqueeze(0)
        indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

        # 6. Denoising loop
        for k in range(num_latent_chunk):
            is_first_chunk = k == 0
            is_second_chunk = k == 1
            if keep_first_frame:
                latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                    :, :, -num_history_latent_frames:
                ].split(history_sizes, dim=2)
                if image_latents is None and is_first_chunk:
                    latents_prefix = torch.zeros(
                        (
                            batch_size,
                            num_channels_latents,
                            1,
                            latents_history_1x.shape[-2],
                            latents_history_1x.shape[-1],
                        ),
                        device=device,
                        dtype=latents_history_1x.dtype,
                    )
                else:
                    latents_prefix = image_latents
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                latents_history_long, latents_history_mid, latents_history_short = history_latents[
                    :, :, -num_history_latent_frames:
                ].split(history_sizes, dim=2)

            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                window_num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            num_inference_steps = (
                sum(pyramid_num_inference_steps_list) * 2
                if is_amplify_first_chunk and self.config.is_distilled and is_first_chunk
                else sum(pyramid_num_inference_steps_list)
            )

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                _, _, _, pyramid_height, pyramid_width = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_latent_frames_per_chunk, num_channels_latents, pyramid_height, pyramid_width
                )
                for _ in range(pyramid_num_stages - 1):
                    pyramid_height //= 2
                    pyramid_width //= 2
                    latents = (
                        F.interpolate(
                            latents,
                            size=(pyramid_height, pyramid_width),
                            mode="bilinear",
                        )
                        * 2
                    )
                latents = latents.reshape(
                    batch_size, num_latent_frames_per_chunk, num_channels_latents, pyramid_height, pyramid_width
                ).permute(0, 2, 1, 3, 4)

                start_point_list = None
                if self.config.is_distilled:
                    start_point_list = [latents]

                for stage_idx in range(pyramid_num_stages):
                    patch_size = self.transformer.config.patch_size
                    image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                        patch_size[0] * patch_size[1] * patch_size[2]
                    )
                    mu = calculate_shift(
                        image_seq_len,
                        self.scheduler.config.get("base_image_seq_len", 256),
                        self.scheduler.config.get("max_image_seq_len", 4096),
                        self.scheduler.config.get("base_shift", 0.5),
                        self.scheduler.config.get("max_shift", 1.15),
                    )
                    self.scheduler.set_timesteps(
                        pyramid_num_inference_steps_list[stage_idx],
                        stage_idx,
                        device=device,
                        mu=mu,
                        is_amplify_first_chunk=is_amplify_first_chunk and is_first_chunk,
                    )
                    timesteps = self.scheduler.timesteps
                    num_warmup_steps = 0
                    self._num_timesteps = len(timesteps)

                    if stage_idx > 0:
                        pyramid_height *= 2
                        pyramid_width *= 2
                        num_frames = latents.shape[2]
                        latents = latents.permute(0, 2, 1, 3, 4).reshape(
                            batch_size * num_latent_frames_per_chunk,
                            num_channels_latents,
                            pyramid_height // 2,
                            pyramid_width // 2,
                        )
                        latents = F.interpolate(latents, size=(pyramid_height, pyramid_width), mode="nearest")
                        latents = latents.reshape(
                            batch_size,
                            num_latent_frames_per_chunk,
                            num_channels_latents,
                            pyramid_height,
                            pyramid_width,
                        ).permute(0, 2, 1, 3, 4)
                        # Fix the stage
                        ori_sigma = 1 - self.scheduler.ori_start_sigmas[stage_idx]  # the original coeff of signal
                        gamma = self.scheduler.config.gamma
                        alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                        beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                        batch_size, channel, num_frames, pyramid_height, pyramid_width = latents.shape
                        noise = self.sample_block_noise(
                            batch_size, channel, num_frames, pyramid_height, pyramid_width, patch_size, device
                        )
                        noise = noise.to(device=device, dtype=transformer_dtype)
                        latents = alpha * latents + beta * noise  # To fix the block artifact

                        if self.config.is_distilled:
                            start_point_list.append(latents)

                    for i, t in enumerate(timesteps):
                        timestep = t.expand(latents.shape[0]).to(torch.int64)

                        latent_model_input = latents.to(transformer_dtype)
                        latents_history_short = latents_history_short.to(transformer_dtype)
                        latents_history_mid = latents_history_mid.to(transformer_dtype)
                        latents_history_long = latents_history_long.to(transformer_dtype)
                        with self.transformer.cache_context("cond"):
                            noise_pred = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                indices_hidden_states=indices_hidden_states,
                                indices_latents_history_short=indices_latents_history_short,
                                indices_latents_history_mid=indices_latents_history_mid,
                                indices_latents_history_long=indices_latents_history_long,
                                latents_history_short=latents_history_short,
                                latents_history_mid=latents_history_mid,
                                latents_history_long=latents_history_long,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]

                        if self.do_classifier_free_guidance:
                            with self.transformer.cache_context("uncond"):
                                noise_uncond = self.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    indices_hidden_states=indices_hidden_states,
                                    indices_latents_history_short=indices_latents_history_short,
                                    indices_latents_history_mid=indices_latents_history_mid,
                                    indices_latents_history_long=indices_latents_history_long,
                                    latents_history_short=latents_history_short,
                                    latents_history_mid=latents_history_mid,
                                    latents_history_long=latents_history_long,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                )[0]

                            if self.config.is_cfg_zero_star:
                                noise_pred_text = noise_pred
                                positive_flat = noise_pred_text.view(batch_size, -1)
                                negative_flat = noise_uncond.view(batch_size, -1)

                                alpha = optimized_scale(positive_flat, negative_flat)
                                alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                                alpha = alpha.to(noise_pred_text.dtype)

                                if (stage_idx == 0 and i <= zero_steps) and use_zero_init:
                                    noise_pred = noise_pred_text * 0.0
                                else:
                                    noise_pred = noise_uncond * alpha + guidance_scale * (
                                        noise_pred_text - noise_uncond * alpha
                                    )
                            else:
                                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                        extra_kwargs = (
                            {
                                "cur_sampling_step": i,
                                "dmd_noisy_tensor": start_point_list[stage_idx]
                                if start_point_list is not None
                                else None,
                                "dmd_sigmas": self.scheduler.sigmas,
                                "dmd_timesteps": self.scheduler.timesteps,
                                "all_timesteps": timesteps,
                            }
                            if self.config.is_distilled
                            else {}
                        )

                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            generator=generator,
                            return_dict=False,
                            **extra_kwargs,
                        )[0]

                        if callback_on_step_end is not None:
                            callback_kwargs = {}
                            for k in callback_on_step_end_tensor_inputs:
                                callback_kwargs[k] = locals()[k]
                            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                            latents = callback_outputs.pop("latents", latents)
                            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                            negative_prompt_embeds = callback_outputs.pop(
                                "negative_prompt_embeds", negative_prompt_embeds
                            )

                        if i == len(timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                        ):
                            progress_bar.update()

                        if XLA_AVAILABLE:
                            xm.mark_step()

                if keep_first_frame and (
                    (is_first_chunk and image_latents is None) or (is_skip_first_chunk and is_second_chunk)
                ):
                    image_latents = latents[:, :, 0:1, :, :]

                total_generated_latent_frames += latents.shape[2]
                history_latents = torch.cat([history_latents, latents], dim=2)
                real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
                current_latents = (
                    real_history_latents[:, :, -num_latent_frames_per_chunk:].to(vae_dtype) / latents_std
                    + latents_mean
                )
                current_video = self.vae.decode(current_latents, return_dict=False)[0]

                if history_video is None:
                    history_video = current_video
                else:
                    history_video = torch.cat([history_video, current_video], dim=2)

        self._current_timestep = None

        if output_type != "latent":
            generated_frames = history_video.size(2)
            generated_frames = (
                generated_frames - 1
            ) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            history_video = history_video[:, :, :generated_frames]
            video = self.video_processor.postprocess_video(history_video, output_type=output_type)
        else:
            video = real_history_latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HeliosPipelineOutput(frames=video)
