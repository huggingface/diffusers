# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
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
import json
import math
from typing import Callable

import numpy as np
import PIL.Image
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLFlux2, BriaFibo2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import BriaFibo2PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# The null caption fibo-2 was trained with for classifier-free guidance: every key of the structured caption, every
# value empty
DEFAULT_NEGATIVE_PROMPT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import json

        >>> import torch
        >>> from diffusers import BriaFibo2Pipeline
        >>> from diffusers.utils import load_image

        >>> # fibo-2 reads structured JSON captions, serialized compactly the way it was trained
        >>> caption = {
        ...     "short_description": "A red bicycle leaning against a white brick wall on a sunny morning.",
        ...     "objects": [{"description": "a red city bicycle with a wicker basket", "location": "center"}],
        ...     "background_setting": "a white brick wall with a green door",
        ...     "lighting": {"conditions": "bright morning sun", "direction": "from the right"},
        ...     "style_medium": "photograph",
        ... }
        >>> prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)

        >>> # turbo: 4 steps without guidance
        >>> pipe = BriaFibo2Pipeline.from_pretrained("briaai/fibo-2-turbo-merge", dtype=torch.bfloat16).to("cuda")
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=1.0).images[0]

        >>> # mopd: 30 steps with guidance 5, the defaults. Each pipeline takes about 25 GB, so free the turbo one first
        >>> del pipe
        >>> pipe = BriaFibo2Pipeline.from_pretrained("briaai/fibo-2-mopd", dtype=torch.bfloat16).to("cuda")
        >>> image = pipe(prompt).images[0]
        >>> image.save("fibo2.png")

        >>> # editing: pass the image, and the instruction as the prompt
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
        ... )
        >>> edited = pipe("Make the background a snowy forest", image=image).images[0]
        >>> edited.save("fibo2_edit.png")
        ```
"""


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


class BriaFibo2Pipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation and image editing with fibo-2.

    Args:
        transformer ([`BriaFibo2Transformer2DModel`]):
            The transformer that denoises the image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler used with `transformer` to denoise the latents.
        vae ([`AutoencoderKLFlux2`]):
            Variational auto-encoder that maps images to latents and back.
        text_encoder (`Qwen3VLForConditionalGeneration`):
            Qwen3-VL. Its hidden states condition the transformer.
        processor (`Qwen3VLProcessor`):
            Turns the prompt, and the images to edit, into Qwen3-VL inputs.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: BriaFibo2Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Qwen3VLForConditionalGeneration,
        processor: Qwen3VLProcessor,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128
        self.tokenizer_max_length = 4096
        self.prompt_template_encode = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # Qwen3-VL layers behind the transformer's six text inputs: the Perceiver's, then one per injection block
        self.text_encoder_out_layers = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
        # Editing: fibo-2 edits up to 5 images at once, and Qwen3-VL sees each one at most 768 pixels a side
        self.max_num_images = 5
        self.text_encoder_image_size = 768

    @staticmethod
    def _clean_caption(caption: dict) -> dict:
        # Training captions hold no empty values, and no aesthetic_score or preference_score: training removes them
        # from the caption and from its sections
        def clean(value):
            if isinstance(value, dict):
                value = {k: clean(v) for k, v in value.items()}
                return {k: v for k, v in value.items() if v not in (None, "", {}, [])}
            if isinstance(value, list):
                return [v for v in map(clean, value) if v not in (None, "", {}, [])]
            return value

        scores = ("aesthetic_score", "preference_score")
        caption = {
            key: {k: v for k, v in value.items() if k not in scores} if isinstance(value, dict) else value
            for key, value in clean(caption).items()
            if key not in scores
        }
        return {key: value for key, value in caption.items() if value != {}}

    def _get_edit_prompt(self, prompt: str, num_images: int) -> str:
        # An edit prompt is one JSON object: a vision marker per image, the caption of the result when the prompt has
        # one, then the edit instruction. The prompt is a JSON caption with an "edit_instruction" key, or the
        # instruction alone
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError:
            caption = None
        caption = dict(caption) if isinstance(caption, dict) else {"edit_instruction": prompt}
        instruction = caption.pop("edit_instruction")
        caption = self._clean_caption(caption)

        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        if num_images == 1:
            edit = {"image": marker}
        else:
            edit = {f"image_{k}": marker for k in range(1, num_images + 1)}
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)

    def _get_qwen_prompt_embeds(
        self, prompt: str | list[str], image: list[PIL.Image.Image] | None, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if image is not None:
            # Every prompt edits the same images
            prompt = [self._get_edit_prompt(p, len(image)) for p in prompt]
            image = image * len(prompt)
        text = [self.prompt_template_encode.format(p) for p in prompt]
        text_inputs = self.processor(
            text=text,
            images=image,
            padding="longest",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        hidden_states = self.text_encoder(**text_inputs, output_hidden_states=True).hidden_states

        # Each layer is scaled to unit RMS per token in float32, then every bundle concatenates its three layers
        dtype = self.text_encoder.dtype
        normalized = {}
        for layer in sorted({layer for layers in self.text_encoder_out_layers for layer in layers}):
            states = hidden_states[layer].float()
            normalized[layer] = (states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(dtype)
        bundles = [
            torch.cat([normalized[layer] for layer in layers], dim=-1) for layers in self.text_encoder_out_layers
        ]
        return torch.stack(bundles, dim=1), text_inputs.attention_mask.bool()

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: list[PIL.Image.Image] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            prompt (`str` or `list[str]`):
                The prompt or prompts to encode. With `image`, each prompt is an edit instruction, or a JSON caption of
                the result with an `"edit_instruction"` key.
            image (`list[PIL.Image.Image]`, *optional*):
                The images to edit, as Qwen3-VL sees them. Every prompt edits all of them.
            device (`torch.device`, *optional*):
                The device to put the embeddings on.
            num_images_per_prompt (`int`, defaults to 1):
                Number of images to generate per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text embeddings of shape `(batch_size, 6, text_len, 7680)`, used instead of `prompt`.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Bool mask of the real tokens in `prompt_embeds`, of shape `(batch_size, text_len)`.

        Returns:
            The text embeddings, and their mask, or `None` when no prompt was padded.
        """
        device = device or self._execution_device
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)

        prompt_embeds = prompt_embeds.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(device).repeat_interleave(num_images_per_prompt, dim=0)
            if prompt_embeds_mask.all():
                # Nothing is padded: attention runs without a mask, which every attention backend supports
                prompt_embeds_mask = None
        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._patchify_latents
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpatchify_latents
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    def _get_image_size(self, aspect_ratio: float) -> tuple[int, int]:
        # The (width, height) fibo-2 was trained at for an aspect ratio: of the sizes divisible by 16, at least 64
        # pixels a side and within 10% of the default 1024x1024 area, the one that crops the least from a picture with
        # that aspect ratio, then the one closest to the default area
        multiple = self.vae_scale_factor * 2
        min_side = 4 * multiple
        area = (self.default_sample_size * self.vae_scale_factor) ** 2
        sizes = [
            (width, height)
            for width in range(min_side, 2 * area // min_side, multiple)
            for height in range(min_side, 2 * area // width, multiple)
            if 0.9 * area <= width * height <= 1.1 * area
        ]

        def crop(size):
            ratio = size[0] / size[1]
            return 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)

        return min(sizes, key=lambda size: (round(crop(size), 9), abs(size[0] * size[1] - area), -size[0]))

    @staticmethod
    def _resize_and_crop(image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        # Scale the image to cover (width, height), keeping its aspect ratio, then crop the center, as in training
        image = image.convert("RGB")
        scale = max(height / image.height, width / image.width)
        resized_width, resized_height = math.ceil(image.width * scale), math.ceil(image.height * scale)
        image = image.resize((resized_width, resized_height), resample=PIL.Image.LANCZOS)
        left, top = (resized_width - width) // 2, (resized_height - height) // 2
        return image.crop((left, top, left + width, top + height))

    def prepare_image_latents(self, image, batch_size, device, dtype):
        # The images to edit are normalized with the VAE's BatchNorm statistics, like the image being generated. The
        # pixels are made contiguous so the VAE's convolutions run the same kernels as in the reference implementation
        image_latents = []
        for img in image:
            img = self.image_processor.preprocess(img).contiguous().to(device=device, dtype=self.vae.dtype)
            latents = self._patchify_latents(self.vae.encode(img).latent_dist.mode().float())
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_scale = (1 / torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps)).view(
                1, -1, 1, 1
            )
            latents = (latents - latents_bn_mean) * latents_bn_scale.to(latents.device, latents.dtype)
            latents = self._unpatchify_latents(latents)
            image_latents.append(latents.repeat(batch_size, 1, 1, 1).to(dtype))
        return image_latents

    def check_inputs(
        self,
        prompt,
        height,
        width,
        image=None,
        mask=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if any(size is not None and size % (self.vae_scale_factor * 2) != 0 for size in (height, width)):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both `prompt` and `prompt_embeds`. Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both `negative_prompt` and `negative_prompt_embeds`. Please make sure to only forward"
                " one of the two."
            )

        if isinstance(prompt, list) and isinstance(negative_prompt, list) and len(prompt) != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt` has {len(negative_prompt)} prompts but `prompt` has {len(prompt)}. Pass one"
                " negative prompt per prompt, or a single string for all of them."
            )

        if image is not None:
            if not all(isinstance(img, PIL.Image.Image) for img in image):
                raise ValueError("`image` has to be a `PIL.Image.Image` or a list of them.")
            if not 1 <= len(image) <= self.max_num_images:
                raise ValueError(f"`image` can hold 1 to {self.max_num_images} images but holds {len(image)}.")
            for p in [] if prompt is None else [prompt] if isinstance(prompt, str) else prompt:
                try:
                    caption = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if isinstance(caption, dict) and "edit_instruction" not in caption:
                    raise ValueError(
                        "To edit an image, `prompt` has to be the edit instruction, or a JSON caption of the result with"
                        f' an "edit_instruction" key, but is {p}'
                    )

        if mask is not None:
            if image is None or len(image) != 1:
                raise ValueError("`mask` needs exactly one image in `image`.")
            if not isinstance(mask, PIL.Image.Image):
                raise ValueError(f"`mask` has to be a `PIL.Image.Image` but is {type(mask)}")
            if mask.size != image[0].size:
                raise ValueError(f"`mask` has size {mask.size} but `image` has size {image[0].size}.")

    def prepare_latents(self, batch_size, num_channels_latents, height, width, device, generator, latents=None):
        if latents is not None:
            return latents.to(device=device, dtype=torch.float32)

        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        return randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

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
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        mask: PIL.Image.Image | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> BriaFibo2PipelineOutput | tuple:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt to guide the image generation, usually a structured JSON prompt. With `image`, the edit
                instruction, or a JSON caption of the result with an `"edit_instruction"` key.
            image (`PIL.Image.Image` or `list[PIL.Image.Image]`, *optional*):
                The image to edit, or up to 5 images to edit together, which the instruction can refer to as
                `<image_1>`, `<image_2>` and so on. Without `image`, the pipeline generates an image from the prompt.
            mask (`PIL.Image.Image`, *optional*):
                The region of `image` to regenerate, white on black, when editing a single image. The region is greyed
                out before the image is encoded.
            height (`int`, *optional*):
                The height in pixels of the generated image. Must be divisible by 16. Defaults to 1024, or when
                editing, to the size fibo-2 was trained at for the aspect ratio of the first image, about one
                megapixel.
            width (`int`, *optional*):
                The width in pixels of the generated image. Must be divisible by 16. Defaults like `height`.
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps. The turbo checkpoint is distilled for 4.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Classifier-free guidance scale. Values above 1 run the negative prompt as a second pass at every step.
                The turbo checkpoint is distilled for 1.0, which turns guidance off.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt to guide away from when guidance is on. Defaults to the null caption fibo-2 was trained
                with: the structured caption with every value empty.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) used to draw the starting noise, for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated starting noise of shape `(batch_size, 32, height // 8, width // 8)`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text embeddings from `encode_prompt`, used instead of `prompt`.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                The mask `encode_prompt` returned with `prompt_embeds`. Needed when they are padded.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text embeddings of the negative prompt, used instead of `negative_prompt`.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                The mask `encode_prompt` returned with `negative_prompt_embeds`. Needed when they are padded.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format: `"pil"`, `"np"`, `"pt"` or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`BriaFibo2PipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step as `callback_on_step_end(self, step, timestep,
                callback_kwargs)`.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*, defaults to `["latents"]`):
                The tensors passed to `callback_on_step_end` in `callback_kwargs`.

        Examples:

        Returns:
            [`BriaFibo2PipelineOutput`] or `tuple`: the generated images.
        """
        if isinstance(image, PIL.Image.Image):
            image = [image]

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            image,
            mask,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        if image is not None and height is None and width is None:
            width, height = self._get_image_size(image[0].width / image[0].height)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._interrupt = False
        device = self._execution_device

        # 2. Prepare the images to edit. A single image is cropped to the size of the result, several images each to
        # the size for their own aspect ratio. The VAE encodes the crops, and Qwen3-VL sees them at most 768 pixels a
        # side
        vae_images = text_encoder_images = None
        if image is not None:
            if mask is not None:
                grey = PIL.Image.new("RGB", image[0].size, (128, 128, 128))
                image = [PIL.Image.composite(grey, image[0].convert("RGB"), mask.convert("L"))]
            if len(image) == 1:
                image_sizes = [(width, height)]
            else:
                image_sizes = [self._get_image_size(img.width / img.height) for img in image]
            vae_images = [self._resize_and_crop(img, *size) for img, size in zip(image, image_sizes)]
            text_encoder_images = [img.copy() for img in vae_images]
            for img in text_encoder_images:
                img.thumbnail((self.text_encoder_image_size, self.text_encoder_image_size))

        # 3. Encode the prompt with the images to edit, and the negative prompt, without them, when guidance is on
        if prompt is not None:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt, text_encoder_images, device, num_images_per_prompt, prompt_embeds, prompt_embeds_mask
        )
        if self.do_classifier_free_guidance:
            if negative_prompt is None and negative_prompt_embeds is None:
                negative_prompt = DEFAULT_NEGATIVE_PROMPT
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
            )

        # 4. Prepare the starting noise, kept in float32 between steps so the updates aren't rounded to the
        # transformer's dtype, and the latents of the images to edit
        latents = self.prepare_latents(
            prompt_embeds.shape[0], self.transformer.config.in_channels, height, width, device, generator, latents
        )
        image_latents = None
        if vae_images is not None:
            image_latents = self.prepare_image_latents(vae_images, latents.shape[0], device, self.transformer.dtype)

        # 5. Prepare timesteps: sigmas from 1 down to 1 / steps, shifted by the image size
        patch_size = self.transformer.config.patch_size
        image_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # the transformer takes the noise level in [0, 1]
                timestep = t.expand(latents.shape[0]) / 1000
                latent_model_input = latents.to(self.transformer.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_embeds_mask,
                    context_latents=image_latents,
                    return_dict=False,
                )[0].float()

                # The negative prompt runs as a second pass, as in the reference implementation: batching it with the
                # prompt would pad both to the longer of the two. The images to edit condition both passes
                if self.do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_embeds_mask,
                        context_latents=image_latents,
                        return_dict=False,
                    )[0].float()
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 7. Decode: undo the latent normalization with the VAE's BatchNorm statistics, then run the VAE decoder
        if output_type == "latent":
            image = latents
        else:
            latents = self._patchify_latents(latents)
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_scale = (1 / torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps)).view(
                1, -1, 1, 1
            )
            latents = latents / latents_bn_scale.to(latents.device, latents.dtype) + latents_bn_mean
            latents = self._unpatchify_latents(latents)
            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return BriaFibo2PipelineOutput(images=image)
