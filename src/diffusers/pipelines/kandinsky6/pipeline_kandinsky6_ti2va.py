# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import copy
import inspect
import math
from collections.abc import Callable

import PIL.Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLHunyuanVideo, Kandinsky6Transformer3DModel, MMAudioVAE, MMAudioVocoder
from ...schedulers import FlowMatchEulerDiscreteScheduler, PiflowScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import Kandinsky6TI2VAPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import Kandinsky6TI2VAPipeline
        >>> from diffusers.utils import encode_video

        >>> pipe = Kandinsky6TI2VAPipeline.from_pretrained(
        ...     "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> output = pipe(
        ...     prompt="A cat and a dog baking a cake together in a kitchen.",
        ...     height=480,
        ...     width=864,
        ...     num_frames=121,
        ...     num_inference_steps=16,
        ...     guidance_scale=1.0,
        ... )
        >>> encode_video(
        ...     output.frames[0],
        ...     fps=24,
        ...     output_path="output.mp4",
        ...     audio=output.audio[0][None],
        ...     audio_sample_rate=pipe.audio_sample_rate,
        ... )
        ```
"""

_PROMPT_TEMPLATE = "\n".join(
    [
        "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
        "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
        "Describe the location of the video, main characters or objects and their action.",
        "Describe the dynamism of the video and presented actions.",
        "Name the visual style of the video: whether it is a professional footage, user generated content, "
        "some kind of animation, video game or scren content.",
        "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
        "Pay attention to the order of key actions shown in the scene.<|im_end|>",
        "<|im_start|>user\n{}<|im_end|>",
    ]
)
# Number of template tokens preceding the user prompt in the Qwen sequence.
_QWEN_CROP_START = 129
_CLIP_MAX_LENGTH = 77
_T2VA_EXPANSION_INSTRUCTION = (
    "You are a prompt beautifier that transforms short user video+audio descriptions into rich, detailed English "
    "prompts specifically optimized for video+audio generation models. Preserve any direct speech between <S> "
    "and <E> exactly as written. If the prompt asks for speech without exact words, add suitable direct speech "
    "between those tags. Put general audio descriptions between <AUDCAP> and <ENDAUDCAP>. Describe the scene, "
    "actions, camera motion, visual style, and audio in detail. Make the prompt dynamic. Answer only with the "
    "expanded prompt.\n\n"
    "Rewrite Prompt: {}"
)
_I2VA_EXPANSION_INSTRUCTION = (
    "You are a prompt beautifier that transforms a short user video+audio description and a provided reference "
    "image into a rich, detailed English prompt optimized for video+audio generation. The reference image is the "
    "ground truth for the initial scene. Keep every visible fact that you mention consistent with it. Do not mainly "
    "describe the image: focus on the requested actions, motion, interactions, camera changes, speech, and audio, "
    "explaining the changes relative to the initial image. Add only details compatible with the reference image; "
    "never invent conflicting objects, identities, colors, locations, or actions. Preserve direct speech between "
    "<S> and <E> exactly. Put general audio descriptions between <AUDCAP> and <ENDAUDCAP>. Make the prompt dynamic "
    "and describe how the scene evolves from the provided image. Answer only with the expanded prompt.\n\n"
    "Rewrite Prompt: {}"
)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
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
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
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
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class Kandinsky6TI2VAPipeline(DiffusionPipeline):
    r"""
    Pipeline for text/image-to-video-and-audio generation with Kandinsky 6.

    Video and audio latents are denoised together by a single multimodal transformer, conditioned on Qwen2.5-VL text
    tokens and a CLIP pooled embedding. An optional reference image conditions the first frame.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`Kandinsky6Transformer3DModel`]):
            Multimodal transformer that denoises the video and audio latents.
        vae ([`AutoencoderKLHunyuanVideo`]):
            Video VAE used to encode the reference image and decode the generated video.
        text_encoder ([`~transformers.Qwen2_5_VLForConditionalGeneration`]):
            Qwen2.5-VL model providing the token-level text embeddings and, optionally, prompt expansion.
        tokenizer ([`~transformers.Qwen2_5_VLProcessor`]):
            Processor of `text_encoder`.
        text_encoder_2 ([`~transformers.CLIPTextModel`]):
            CLIP text encoder providing the pooled text embedding.
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):
            Tokenizer of `text_encoder_2`.
        scheduler ([`FlowMatchEulerDiscreteScheduler`] or [`PiflowScheduler`]):
            Scheduler used with `transformer` to denoise the latents. Distilled checkpoints ship with a
            [`PiflowScheduler`] and must be run with `guidance_scale=1.0`.
        audio_vae ([`MMAudioVAE`], *optional*):
            Audio VAE used to decode the generated audio latents into a mel spectrogram. Only needed when
            `sample_audio=True`.
        vocoder ([`MMAudioVocoder`], *optional*):
            Vocoder used to turn the mel spectrogram `audio_vae` decodes into a waveform. Only needed when
            `sample_audio=True`.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae->audio_vae->vocoder"
    _optional_components = ["audio_vae", "vocoder"]
    _callback_tensor_inputs = ["latents", "audio_latents", "prompt_embeds", "negative_prompt_embeds"]
    _DEFAULT_NEGATIVE_PROMPT = (
        "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
        "worst quality, low quality, ugly, deformed, walking backwards"
    )

    def __init__(
        self,
        transformer: Kandinsky6Transformer3DModel,
        vae: AutoencoderKLHunyuanVideo,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2_5_VLProcessor,
        text_encoder_2: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler | PiflowScheduler,
        audio_vae: MMAudioVAE | None = None,
        vocoder: MMAudioVocoder | None = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            audio_vae=audio_vae,
            vocoder=vocoder,
        )

        self.vae_scale_factor_spatial = (
            self.vae.config.spatial_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if getattr(self, "vae", None) is not None else 4
        )
        self.transformer_patch_size = (
            tuple(self.transformer.config.patch_size) if getattr(self, "transformer", None) is not None else (1, 2, 2)
        )
        self.audio_sample_rate = (
            self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 44_100
        )
        self.audio_latent_hop_length = (
            self.audio_vae.latent_hop_length if getattr(self, "audio_vae", None) is not None else 1024
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @staticmethod
    def _get_prompt_embeds(
        prompt: list[str],
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode prompts with Qwen2.5-VL (token embeddings) and CLIP (pooled embedding).

        Returns the token embeddings, the pooled embeddings and a boolean padding mask, which is `None` when no prompt
        in the batch is padded (a mask without padding carries no information, and dropping it keeps every attention
        backend available).
        """
        inputs = tokenizer(
            text=[_PROMPT_TEMPLATE.format(item) for item in prompt],
            images=None,
            videos=None,
            max_length=max_sequence_length + _QWEN_CROP_START,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(device)
        qwen_output = text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )
        prompt_embeds = qwen_output["hidden_states"][-1][:, _QWEN_CROP_START:]
        prompt_attention_mask = inputs["attention_mask"][:, _QWEN_CROP_START:].to(dtype=torch.bool)
        if prompt_attention_mask.all():
            prompt_attention_mask = None

        clip_inputs = tokenizer_2(
            prompt,
            max_length=_CLIP_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        pooled_prompt_embeds = text_encoder_2(**clip_inputs)["pooler_output"]
        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`):
                Prompt to be encoded.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt not to guide the generation. Ignored when `do_classifier_free_guidance` is `False`.
            do_classifier_free_guidance (`bool`, defaults to `True`):
                Whether to also encode the negative prompt.
            num_videos_per_prompt (`int`, defaults to `1`):
                Number of videos generated per prompt; the embeddings are repeated accordingly.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated Qwen2.5-VL text embeddings. Skips encoding `prompt`.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated CLIP pooled text embeddings. Must be given together with `prompt_embeds`.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Boolean padding mask of `prompt_embeds`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative Qwen2.5-VL text embeddings.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative CLIP pooled text embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Boolean padding mask of `negative_prompt_embeds`.
            max_sequence_length (`int`, defaults to `1024`):
                Maximum number of prompt tokens after the chat template.
            device (`torch.device`, *optional*):
                Device to run the text encoders on.
            dtype (`torch.dtype`, *optional*):
                Dtype of the returned embeddings.
        """
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self._get_prompt_embeds(
                prompt,
                self.tokenizer,
                self.text_encoder,
                self.tokenizer_2,
                self.text_encoder_2,
                max_sequence_length,
                device,
            )
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or self._DEFAULT_NEGATIVE_PROMPT
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt = negative_prompt * len(prompt) if len(negative_prompt) == 1 else negative_prompt
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = (
                self._get_prompt_embeds(
                    negative_prompt,
                    self.tokenizer,
                    self.text_encoder,
                    self.tokenizer_2,
                    self.text_encoder_2,
                    max_sequence_length,
                    device,
                )
            )

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype).repeat_interleave(num_videos_per_prompt, dim=0)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype).repeat_interleave(
            num_videos_per_prompt, dim=0
        )
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(device).repeat_interleave(num_videos_per_prompt, dim=0)
        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype).repeat_interleave(
                num_videos_per_prompt, dim=0
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                device=device, dtype=dtype
            ).repeat_interleave(num_videos_per_prompt, dim=0)
            if negative_prompt_attention_mask is not None:
                negative_prompt_attention_mask = negative_prompt_attention_mask.to(device).repeat_interleave(
                    num_videos_per_prompt, dim=0
                )

        return (
            prompt_embeds,
            pooled_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_attention_mask,
        )

    @staticmethod
    def expand_prompts(
        prompt: str | list[str],
        tokenizer,
        text_encoder,
        device: torch.device,
        image: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        max_sequence_length: int = 1024,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> str | list[str]:
        r"""
        Rewrites short prompts into detailed video+audio prompts with the Qwen2.5-VL text encoder, grounding them on
        the reference image when one is given. A `staticmethod` so it can be used standalone, before running the
        pipeline.

        Args:
            prompt (`str` or `list[str]`):
                Prompt or prompts to expand.
            tokenizer:
                The Qwen2.5-VL processor, e.g. `pipe.tokenizer`.
            text_encoder:
                The Qwen2.5-VL model, e.g. `pipe.text_encoder`.
            device (`torch.device`):
                Device to run the text encoder on.
            image (`PIL.Image.Image` or `list[PIL.Image.Image]`, *optional*):
                Reference image(s) of an image-to-video call.
            max_sequence_length (`int`, defaults to `1024`):
                Maximum number of generated tokens per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Seeds the sampled expansion; a list must match `prompt`'s length, one generator per item. `generate`
                draws from the global RNG, so the global RNG is seeded from this generator's seed; later `randn_tensor`
                calls keep using `generator` directly.

        Returns:
            `str` or `list[str]`: The expanded prompt(s).
        """
        if isinstance(prompt, list):
            images = image if isinstance(image, list) else [image] * len(prompt)
            generators = generator if isinstance(generator, list) else [generator] * len(prompt)
            return [
                Kandinsky6TI2VAPipeline.expand_prompts(
                    item,
                    tokenizer,
                    text_encoder,
                    device,
                    image=item_image,
                    max_sequence_length=max_sequence_length,
                    generator=item_generator,
                )
                for item, item_image, item_generator in zip(prompt, images, generators, strict=True)
            ]
        if image is not None and not isinstance(image, PIL.Image.Image):
            raise ValueError("`expand_prompts` expects `image` as a `PIL.Image.Image`")

        instruction = (_I2VA_EXPANSION_INSTRUCTION if image is not None else _T2VA_EXPANSION_INSTRUCTION).format(
            prompt
        )
        content = [{"type": "image", "image": image}] if image is not None else []
        content.append({"type": "text", "text": instruction})
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            text=[text],
            images=[image] if image is not None else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(device)
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        generated = text_encoder.generate(**inputs, max_new_tokens=max_sequence_length)
        generated = generated[:, inputs["input_ids"].shape[1] :]
        return tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        image,
        sample_audio,
        expand_prompts,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        spatial_multiple = self.vae_scale_factor_spatial * max(self.transformer_patch_size[1:])
        if height % spatial_multiple != 0 or width % spatial_multiple != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {spatial_multiple} but are {height} and {width}."
            )
        if num_frames < 1:
            raise ValueError(f"`num_frames` has to be positive but is {num_frames}.")

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
        if expand_prompts and prompt_embeds is not None:
            raise ValueError("`expand_prompts=True` requires `prompt`; it cannot be used with `prompt_embeds`.")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        if (prompt_embeds is None) != (pooled_prompt_embeds is None):
            raise ValueError("`prompt_embeds` and `pooled_prompt_embeds` must be provided together.")
        if (negative_prompt_embeds is None) != (negative_pooled_prompt_embeds is None):
            raise ValueError("`negative_prompt_embeds` and `negative_pooled_prompt_embeds` must be provided together.")

        if sample_audio and (getattr(self, "audio_vae", None) is None or getattr(self, "vocoder", None) is None):
            raise ValueError("`sample_audio=True` requires an `audio_vae` and a `vocoder`.")
        if image is not None:
            if not self.transformer.config.visual_cond:
                raise ValueError("Image conditioning requires a transformer with `visual_cond=True`.")
            if self.transformer.config.visual_token_type_num_embeddings < 2:
                raise ValueError(
                    "Image conditioning requires a transformer with `visual_token_type_num_embeddings >= 2`."
                )

    def encode_image(
        self,
        image: PipelineImageInput,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        num_videos_per_prompt: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""
        Encodes the reference image(s) into first-frame latents of shape `(batch_size, latent_height, latent_width,
        latent_channels)`, scaled by the VAE `scaling_factor`. PIL images are resized and center-cropped to `height x
        width`; tensors and arrays must already have that size. The latents are repeated `num_videos_per_prompt` times
        along the batch dimension.
        """
        is_pil = isinstance(image, PIL.Image.Image) or (
            isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        )
        image = self.video_processor.preprocess(
            image, height=height, width=width, resize_mode="crop" if is_pil else "default"
        )
        image = image.to(device=device, dtype=self.vae.dtype).unsqueeze(2)
        latents = retrieve_latents(self.vae.encode(image), generator=generator) * self.vae.config.scaling_factor
        latents = latents[:, :, 0].permute(0, 2, 3, 1).to(dtype)
        return latents.repeat_interleave(num_videos_per_prompt, dim=0)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Returns video latents in the transformer's `(batch_size, num_frames, height, width, channels)` layout.
        A user-provided `latents` tensor is expected in the `(batch_size, channels, num_frames, height, width)` layout.
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        if latents is not None:
            return latents.to(device=device, dtype=dtype).permute(0, 2, 3, 4, 1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        shape = (batch_size, num_latent_frames, latent_height, latent_width, num_channels_latents)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def prepare_audio_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        audio_length: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        audio_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Returns audio latents in the transformer's `(batch_size, audio_length, channels)` layout. A user-provided
        `audio_latents` tensor is expected in the `(batch_size, channels, audio_length)` layout."""
        if audio_latents is not None:
            return audio_latents.to(device=device, dtype=dtype).permute(0, 2, 1)
        shape = (batch_size, audio_length, num_channels_latents)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

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
        prompt: str | list[str] | None = None,
        image: PipelineImageInput | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        sample_audio: bool = True,
        expand_prompts: bool = False,
        max_sequence_length: int = 1024,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> Kandinsky6TI2VAPipelineOutput | tuple:
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the generation. Required unless `prompt_embeds` is given.
            image (`PipelineImageInput`, *optional*):
                Reference image(s) conditioning the first frame (image-to-video-and-audio).
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the generation. Defaults to the Kandinsky 6 negative prompt.
            height (`int`, defaults to `512`):
                Height of the generated video in pixels.
            width (`int`, defaults to `768`):
                Width of the generated video in pixels.
            num_frames (`int`, defaults to `121`):
                Number of generated frames.
            frame_rate (`float`, defaults to `24.0`):
                Frame rate the video is generated at; sets the length of the synchronized audio.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. Use `16` with the distilled checkpoints.
            timesteps (`list[int]`, *optional*):
                Custom timesteps for schedulers that support them.
            sigmas (`list[float]`, *optional*):
                Custom sigmas for schedulers that support them.
            guidance_scale (`float`, defaults to `5.0`):
                Classifier-free guidance scale. Must be `1.0` with a [`PiflowScheduler`].
            num_videos_per_prompt (`int`, defaults to `1`):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) used for the initial noise and the reference image encoding.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video latents of shape `(batch_size, channels, num_latent_frames, latent_height,
                latent_width)`.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio latents of shape `(batch_size, channels, audio_length)`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated Qwen2.5-VL text embeddings.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated CLIP pooled text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative Qwen2.5-VL text embeddings.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative CLIP pooled text embeddings.
            sample_audio (`bool`, defaults to `True`):
                Whether to generate synchronized audio. Requires the pipeline to have an `audio_vae` and a `vocoder`.
            expand_prompts (`bool`, defaults to `False`):
                Whether to rewrite the prompts with [`~Kandinsky6TI2VAPipeline.expand_prompts`] before encoding.
            max_sequence_length (`int`, defaults to `1024`):
                Maximum number of prompt tokens after the chat template.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generated video: `"pil"`, `"np"`, `"pt"` or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`Kandinsky6TI2VAPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step with `callback_on_step_end(self, step, timestep,
                callback_kwargs)`. It may return a dict overriding the listed tensors.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                Tensor inputs passed to `callback_on_step_end`; a subset of `_callback_tensor_inputs`.

        Examples:

        Returns:
            [`Kandinsky6TI2VAPipelineOutput`] or `tuple`:
                The generated video and audio; a `(frames, audio)` tuple when `return_dict=False`.
        """
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            image=image,
            sample_audio=sample_audio,
            expand_prompts=expand_prompts,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]
        device = self._execution_device
        dtype = self.transformer.dtype

        # 3. Encode input prompt
        if expand_prompts:
            prompt = self.expand_prompts(
                prompt,
                self.tokenizer,
                self.text_encoder,
                device,
                image=image,
                max_sequence_length=max_sequence_length,
                generator=generator,
            )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        # 4. Encode the reference image
        first_frame_latents = None
        if image is not None:
            first_frame_latents = self.encode_image(
                image, height, width, device, dtype, num_videos_per_prompt, generator
            )

        # 5. Prepare timesteps. Audio uses a second scheduler instance so that both modalities keep their own step
        # counter.
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        audio_scheduler = copy.deepcopy(self.scheduler) if sample_audio else None
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        batch_size = batch_size * num_videos_per_prompt
        latents = self.prepare_latents(
            batch_size,
            self.transformer.config.in_visual_dim,
            height,
            width,
            num_frames,
            dtype,
            device,
            generator,
            latents,
        )
        num_latent_frames = latents.shape[1]
        if sample_audio:
            audio_length = math.ceil(
                ((num_latent_frames - 1) * self.vae_scale_factor_temporal + 1)
                / frame_rate
                * self.audio_sample_rate
                / self.audio_latent_hop_length
            )
            audio_latents = self.prepare_audio_latents(
                batch_size, self.transformer.config.in_audio_dim, audio_length, dtype, device, generator, audio_latents
            )
        else:
            audio_latents = None

        # Image conditioning appends the clean reference frame as an extra, masked frame that reuses the first
        # frame's rotary position and carries token type `1`.
        tail_cond = image is not None
        visual_token_type_ids = None
        visual_rope_pos = None
        if tail_cond:
            latents = torch.cat([latents, first_frame_latents[:, None]], dim=1)
            visual_token_type_ids = torch.zeros((batch_size, num_latent_frames + 1), dtype=torch.long, device=device)
            visual_token_type_ids[:, -1] = 1
            patch_t, patch_h, patch_w = self.transformer_patch_size
            visual_rope_pos = (
                torch.cat(
                    [
                        torch.arange(num_latent_frames // patch_t, device=device),
                        torch.zeros(1, dtype=torch.long, device=device),
                    ]
                ),
                torch.arange(latents.shape[2] // patch_h, device=device),
                torch.arange(latents.shape[3] // patch_w, device=device),
            )

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep = t.expand(batch_size)

                # Visual conditioning channels: [latent | conditioning latent | mask]. The conditioning-latent
                # channel is unused (always zero) and only kept to match the transformer's fixed input width; the
                # reference frame itself is appended as an extra, masked tail frame (see `tail_cond` above).
                latent_model_input = latents
                if self.transformer.config.visual_cond:
                    cond_latents = torch.zeros_like(latents)
                    cond_mask = torch.zeros((*latents.shape[:-1], 1), dtype=latents.dtype, device=device)
                    if first_frame_latents is not None:
                        latents[:, -1] = first_frame_latents
                        cond_mask[:, -1] = 1
                    latent_model_input = torch.cat([latents, cond_latents, cond_mask], dim=-1)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        audio_hidden_states=audio_latents,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timestep,
                        visual_rope_pos=visual_rope_pos,
                        encoder_attention_mask=prompt_attention_mask,
                        visual_token_type_ids=visual_token_type_ids,
                        return_dict=False,
                    )
                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            audio_hidden_states=audio_latents,
                            encoder_hidden_states=negative_prompt_embeds,
                            pooled_projections=negative_pooled_prompt_embeds,
                            timestep=timestep,
                            visual_rope_pos=visual_rope_pos,
                            encoder_attention_mask=negative_prompt_attention_mask,
                            visual_token_type_ids=visual_token_type_ids,
                            return_dict=False,
                        )
                    noise_pred = tuple(
                        uncond + self.guidance_scale * (cond - uncond)
                        for cond, uncond in zip(noise_pred, noise_pred_uncond)
                    )

                latents = self.scheduler.step(noise_pred[0], t, latents, return_dict=False)[0]
                if sample_audio:
                    audio_latents = audio_scheduler.step(noise_pred[1], t, audio_latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    audio_latents = callback_outputs.pop("audio_latents", audio_latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.update()

        self._current_timestep = None

        # 8. Drop the appended tail frame used for reference-image conditioning
        if tail_cond:
            latents = latents[:, :-1]

        # 9. Decode
        latents = latents.permute(0, 4, 1, 2, 3)
        audio_latents = audio_latents.permute(0, 2, 1) if sample_audio else None
        if output_type == "latent":
            video, audio = latents, audio_latents
        else:
            video = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
            audio = None
            if sample_audio:
                audio_latents = audio_latents.to(self.audio_vae.dtype) / self.audio_vae.config.scaling_factor
                mel = self.audio_vae.decode(audio_latents, return_dict=False)[0]
                audio = self.vocoder(mel.to(self.vocoder.dtype), return_dict=False)[0][:, 0].float()
                if output_type == "np":
                    audio = audio.cpu().numpy()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, audio)
        return Kandinsky6TI2VAPipelineOutput(frames=video, audio=audio)
