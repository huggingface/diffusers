# Copyright 2025 The SkyReels-V2 Team, The Wan Team and The HuggingFace Team. All rights reserved.
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
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ftfy
import PIL.Image
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import WanLoraLoaderMixin
from ...models import AutoencoderKLWan, SkyReelsV2Transformer3DModel
from ...schedulers import FlowMatchUniPCMultistepScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import SkyReelsV2PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """\
    Examples:
        ```py
        >>> import torch
        >>> import PIL.Image
        >>> from diffusers import SkyReelsV2DiffusionForcingImageToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> # Load the pipeline
        >>> pipe = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(
        ...     "HF_placeholder/SkyReels-V2-DF-1.3B-540P", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # TODO
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


class SkyReelsV2DiffusionForcingImageToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    """
    Pipeline for video generation with diffusion forcing using SkyReels-V2. This pipeline supports two main tasks:
    Text-to-Video (t2v) and Image-to-Video (i2v)

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a specific device, etc.).

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`UMT5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`SkyReelsV2Transformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchUniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: SkyReelsV2Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchUniPCMultistepScheduler,
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

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
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
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
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

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
        )

        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

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
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
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
        max_sequence_length: int = 512, # TODO: check
        overlap_history: Optional[int] = None,
        shift: float = 1.0,
        addnoise_condition: float = 0.0,
        base_num_frames: int = 97,
        ar_step: int = 5,
        causal_block_size: Optional[int] = None,
        fps: int = 24,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `97`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
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
                Whether or not to return a [`SkyReelsV2PipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int`, *optional*, defaults to `512`):
                The maximum sequence length of the prompt.
            shift (`float`, *optional*, defaults to `5.0`):
                The shift of the flow.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.
        Examples:

        Returns:
            [`~SkyReelsV2PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`SkyReelsV2PipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        latent_height = height // 8
        latent_width = width // 8
        latent_length = (num_frames - 1) // 4 + 1

        if causal_block_size is None:
            causal_block_size = self.transformer.num_frame_per_block
        fps_embeds = [fps] * prompt_embeds.shape[0]
        fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        #prompt_embeds = self.text_encoder.encode(prompt).to(self.transformer.dtype)
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
        #prefix_video, predix_video_latent_length = self.encode_image(image, height, width, num_frames)
        if image_embeds is None:
            image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        timesteps = self.scheduler.timesteps

        if overlap_history is None or base_num_frames is None or num_frames <= base_num_frames:
            # short video generation
            latent_shape = [16, latent_length, latent_height, latent_width]
            latents = self.prepare_latents(
                latent_shape, dtype=transformer_dtype, device=prompt_embeds.device, generator=generator
            )
            latents = [latents]
            if prefix_video is not None:
                latents[0][:, :predix_video_latent_length] = prefix_video[0].to(transformer_dtype)
            base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_length
            step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                latent_length, timesteps, base_num_frames, ar_step, predix_video_latent_length, causal_block_size
            )
            sample_schedulers = []
            for _ in range(latent_length):
                sample_scheduler = FlowMatchUniPCMultistepScheduler(
                    num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
                )
                sample_scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.device, shift=shift)
                sample_schedulers.append(sample_scheduler)
            sample_schedulers_counter = [0] * latent_length
            self.transformer.to(self.device)
            for i, timestep_i in enumerate(tqdm(step_matrix)):
                update_mask_i = step_update_mask[i]
                valid_interval_i = valid_interval[i]
                valid_interval_start, valid_interval_end = valid_interval_i
                timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                latent_model_input = [latents[0][:, valid_interval_start:valid_interval_end, :, :].clone()]
                if addnoise_condition > 0 and valid_interval_start < predix_video_latent_length:
                    noise_factor = 0.001 * addnoise_condition
                    timestep_for_noised_condition = addnoise_condition
                    latent_model_input[0][:, valid_interval_start:predix_video_latent_length] = (
                        latent_model_input[0][:, valid_interval_start:predix_video_latent_length] * (1.0 - noise_factor)
                        + torch.randn_like(latent_model_input[0][:, valid_interval_start:predix_video_latent_length])
                        * noise_factor
                    )
                    timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
                if not self.do_classifier_free_guidance:
                    noise_pred = self.transformer(
                        torch.stack([latent_model_input[0]]),
                        t=timestep,
                        context=prompt_embeds,
                        fps=fps_embeds,
                    )[0]
                else:
                    noise_pred_cond = self.transformer(
                        torch.stack([latent_model_input[0]]),
                        t=timestep,
                        context=prompt_embeds,
                        fps=fps_embeds,
                    )[0]
                    noise_pred_uncond = self.transformer(
                        torch.stack([latent_model_input[0]]),
                        t=timestep,
                        context=negative_prompt_embeds,
                        fps=fps_embeds,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                for idx in range(valid_interval_start, valid_interval_end):
                    if update_mask_i[idx].item():
                        latents[0][:, idx] = sample_schedulers[idx].step(
                            noise_pred[:, idx - valid_interval_start],
                            timestep_i[idx],
                            latents[0][:, idx],
                            return_dict=False,
                            generator=generator,
                        )[0]
                        sample_schedulers_counter[idx] += 1
            if self.offload:
                self.transformer.cpu()
                torch.cuda.empty_cache()
            x0 = latents[0].unsqueeze(0)
            videos = self.vae.decode(x0)
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = [video for video in videos]
            videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
            video = [video.cpu().numpy().astype(np.uint8) for video in videos]
        else:
            # long video generation
            base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_length
            overlap_history_frames = (overlap_history - 1) // 4 + 1
            n_iter = 1 + (latent_length - base_num_frames - 1) // (base_num_frames - overlap_history_frames) + 1
            print(f"n_iter:{n_iter}")
            output_video = None
            for i in range(n_iter):
                if output_video is not None:  # i !=0
                    prefix_video = output_video[:, -overlap_history:].to(prompt_embeds.device)
                    prefix_video = [self.vae.encode(prefix_video.unsqueeze(0))[0]]  # [(c, f, h, w)]
                    if prefix_video[0].shape[1] % causal_block_size != 0:
                        truncate_len = prefix_video[0].shape[1] % causal_block_size
                        print("the length of prefix video is truncated for the casual block size alignment.")
                        prefix_video[0] = prefix_video[0][:, : prefix_video[0].shape[1] - truncate_len]
                    predix_video_latent_length = prefix_video[0].shape[1]
                    finished_frame_num = i * (base_num_frames - overlap_history_frames) + overlap_history_frames
                    left_frame_num = latent_length - finished_frame_num
                    base_num_frames_iter = min(left_frame_num + overlap_history_frames, base_num_frames)
                    if ar_step > 0 and self.transformer.enable_teacache:
                        num_steps = num_inference_steps + ((base_num_frames_iter - overlap_history_frames) // causal_block_size - 1) * ar_step
                        self.transformer.num_steps = num_steps
                else:  # i == 0
                    base_num_frames_iter = base_num_frames
                latent_shape = [16, base_num_frames_iter, latent_height, latent_width]
                latents = self.prepare_latents(
                    latent_shape, dtype=transformer_dtype, device=prompt_embeds.device, generator=generator
                )
                latents = [latents]
                if prefix_video is not None:
                    latents[0][:, :predix_video_latent_length] = prefix_video[0].to(transformer_dtype)
                step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                    base_num_frames_iter,
                    timesteps,
                    base_num_frames_iter,
                    ar_step,
                    predix_video_latent_length,
                    causal_block_size,
                )
                sample_schedulers = []
                for _ in range(base_num_frames_iter):
                    sample_scheduler = FlowMatchUniPCMultistepScheduler(
                        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
                    )
                    sample_scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                sample_schedulers_counter = [0] * base_num_frames_iter
                self.transformer.to(self.device)
                for i, timestep_i in enumerate(tqdm(step_matrix)):
                    update_mask_i = step_update_mask[i]
                    valid_interval_i = valid_interval[i]
                    valid_interval_start, valid_interval_end = valid_interval_i
                    timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                    latent_model_input = [latents[0][:, valid_interval_start:valid_interval_end, :, :].clone()]
                    if addnoise_condition > 0 and valid_interval_start < predix_video_latent_length:
                        noise_factor = 0.001 * addnoise_condition
                        timestep_for_noised_condition = addnoise_condition
                        latent_model_input[0][:, valid_interval_start:predix_video_latent_length] = (
                            latent_model_input[0][:, valid_interval_start:predix_video_latent_length]
                            * (1.0 - noise_factor)
                            + torch.randn_like(
                                latent_model_input[0][:, valid_interval_start:predix_video_latent_length]
                            )
                            * noise_factor
                        )
                        timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
                    if not self.do_classifier_free_guidance:
                        noise_pred = self.transformer(
                            torch.stack([latent_model_input[0]]),
                            t=timestep,
                            context=prompt_embeds,
                            fps=fps_embeds,
                        )[0]
                    else:
                        noise_pred_cond = self.transformer(
                            torch.stack([latent_model_input[0]]),
                            t=timestep,
                            context=prompt_embeds,
                            fps=fps_embeds,
                        )[0]
                        noise_pred_uncond = self.transformer(
                            torch.stack([latent_model_input[0]]),
                            t=timestep,
                            context=negative_prompt_embeds,
                            fps=fps_embeds,
                        )[0]
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    for idx in range(valid_interval_start, valid_interval_end):
                        if update_mask_i[idx].item():
                            latents[0][:, idx] = sample_schedulers[idx].step(
                                noise_pred[:, idx - valid_interval_start],
                                timestep_i[idx],
                                latents[0][:, idx],
                                return_dict=False,
                                generator=generator,
                            )[0]
                            sample_schedulers_counter[idx] += 1
                if self.offload:
                    self.transformer.cpu()
                    torch.cuda.empty_cache()
                x0 = latents[0].unsqueeze(0)
                videos = [self.vae.decode(x0)[0]]
                if output_video is None:
                    output_video = videos[0].clamp(-1, 1).cpu()  # c, f, h, w
                else:
                    output_video = torch.cat(
                        [output_video, videos[0][:, overlap_history:].clamp(-1, 1).cpu()], 1
                    )  # c, f, h, w
            output_video = [(output_video / 2 + 0.5).clamp(0, 1)]
            output_video = [video for video in output_video]
            output_video = [video.permute(1, 2, 3, 0) * 255 for video in output_video]
            video = [video.cpu().numpy().astype(np.uint8) for video in output_video]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return SkyReelsV2PipelineOutput(frames=video)
