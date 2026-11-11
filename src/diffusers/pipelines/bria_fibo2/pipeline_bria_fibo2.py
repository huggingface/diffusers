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

from ...loaders import BriaFibo2LoraLoaderMixin
from ...models import AutoencoderKLFlux2, BriaFibo2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import BriaFibo2ImageProcessor
from .pipeline_output import BriaFibo2PipelineOutput
from .text_images import draw_text_image, load_font


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# The null caption Fibo 2.0 was trained with for classifier-free guidance: every key of the structured caption, every
# value empty
DEFAULT_NEGATIVE_PROMPT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)

# Where an image stands in a Qwen3-VL prompt
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

# The chat turn each prompt is wrapped in for Qwen3-VL
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

# Qwen3-VL layers behind the transformer's six text inputs: the Perceiver's, then one per injection block
TEXT_ENCODER_OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import json

        >>> import torch
        >>> from diffusers import BriaFibo2Pipeline
        >>> from diffusers.utils import load_image

        >>> # Fibo 2.0 reads structured JSON captions, serialized compactly the way it was trained
        >>> caption = {
        ...     "short_description": "A red bicycle leaning against a white brick wall on a sunny morning.",
        ...     "objects": [{"description": "a red city bicycle with a wicker basket", "location": "center"}],
        ...     "background_setting": "a white brick wall with a green door",
        ...     "lighting": {"conditions": "bright morning sun", "direction": "from the right"},
        ...     "style_medium": "photograph",
        ... }
        >>> prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)

        >>> # 30 steps with guidance 5, the defaults
        >>> pipe = BriaFibo2Pipeline.from_pretrained("briaai/Fibo-2.0-dev", dtype=torch.bfloat16).to("cuda")
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


class BriaFibo2Pipeline(DiffusionPipeline, BriaFibo2LoraLoaderMixin):
    r"""
    Pipeline for text-to-image generation and image editing with Fibo 2.0.

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
        self.image_processor = BriaFibo2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128
        self.tokenizer_max_length = 4096
        self.prompt_template_encode = PROMPT_TEMPLATE
        self.text_encoder_out_layers = TEXT_ENCODER_OUT_LAYERS
        # Editing: Fibo 2.0 edits up to 5 images at once, and Qwen3-VL sees each one at most 768 pixels a side
        self.max_num_images = 5
        self.text_encoder_image_size = 768
        # Images of the caption's text: a generation shows them to Qwen3-VL only once the noise level is at or below
        # this, as research renders; an edit shows them at every step
        self.text_images_max_noise_level = 0.9

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[dict | str | None, str | None]:
        # A prompt's caption, in the form Fibo 2.0 was trained with, and its edit instruction. A prompt that isn't a JSON
        # object comes back as it is, with no instruction: a short description, or an edit's instruction alone. An edit
        # prompt's caption is its other keys or its "structured_caption" key, a JSON caption or a short description
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError:
            return prompt, None
        if not isinstance(caption, dict):
            return prompt, None
        instruction = caption.pop("edit_instruction", None)
        if "structured_caption" in caption:
            caption = caption.pop("structured_caption")
            if not isinstance(caption, dict):
                return caption or None, instruction

        # Training captions hold no empty values (None, "", {}, [] or NaN), and no aesthetic_score or preference_score:
        # training removes them from the caption and from its sections
        def empty(value):
            if isinstance(value, float):
                return math.isnan(value)
            return value is None or (isinstance(value, (str, dict, list)) and not value)

        def clean(value):
            if isinstance(value, dict):
                value = {k: clean(v) for k, v in value.items()}
                return {k: v for k, v in value.items() if not empty(v)}
            if isinstance(value, list):
                return [v for v in map(clean, value) if not empty(v)]
            return value

        keys = set(caption)
        scores = ("aesthetic_score", "preference_score")
        caption = {
            key: {k: v for k, v in value.items() if k not in scores} if isinstance(value, dict) else value
            for key, value in clean(caption).items()
            if key not in scores
        }
        caption = {key: value for key, value in caption.items() if value != {}}
        # A caption of exactly a "short_description" and a "text_render" list keeps those keys in that order, and the
        # list even when it's empty: it then says the image has no text
        if keys == {"short_description", "text_render"}:
            caption = {
                "short_description": caption.get("short_description") or "",
                "text_render": caption.get("text_render") or [],
            }
        return caption or None, instruction

    @staticmethod
    def _mark_text_images(caption: dict, marker: str) -> dict:
        # Marks, in place, each text_render entry that has text with an "image_tokens" entry: the vision marker of the
        # image of its text, or empty where the image isn't shown
        for entry in caption.get("text_render") or []:
            if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]:
                entry["image_tokens"] = marker
        return caption

    @classmethod
    def _get_edit_prompt(cls, prompt: str, num_images: int, text_images: bool = False) -> str:
        # An edit prompt is one JSON object: a vision marker per image, the caption of the result when the prompt has
        # one, then the edit instruction. With images of the caption's text, each text entry gets the marker of its
        # image, which follows the images to edit
        caption, instruction = cls._parse_prompt(prompt)
        if instruction is None:
            # The prompt is the instruction alone
            caption, instruction = None, prompt
        if text_images and isinstance(caption, dict):
            cls._mark_text_images(caption, VISION_MARKER)

        marker = VISION_MARKER
        if num_images == 1:
            edit = {"image": marker}
        else:
            edit = {f"image_{k}": marker for k in range(1, num_images + 1)}
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _get_qwen_prompt_embeds(
        text_encoder: Qwen3VLForConditionalGeneration,
        processor: Qwen3VLProcessor,
        prompt: str | list[str],
        image: list[PIL.Image.Image] | None = None,
        text_images: list[list[PIL.Image.Image]] | None = None,
        device: torch.device | None = None,
        max_sequence_length: int = 4096,
        prompt_template: str = PROMPT_TEMPLATE,
        hidden_states_layers: tuple[tuple[int, ...], ...] = TEXT_ENCODER_OUT_LAYERS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or text_encoder.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_images = text_images or [[]] * len(prompt)
        if len(text_images) != len(prompt):
            raise ValueError(
                f"`text_images` holds images for {len(text_images)} prompts, but there are {len(prompt)}."
            )
        # Qwen3-VL takes the batch's images in the order of their markers: each prompt's images to edit, which every
        # prompt edits, then the images of its caption's text
        text, images = [], []
        for p, own_text_images in zip(prompt, text_images):
            if image is not None:
                p = BriaFibo2Pipeline._get_edit_prompt(p, len(image), text_images=bool(own_text_images))
                images += image
            elif own_text_images:
                caption = BriaFibo2Pipeline._mark_text_images(json.loads(p), VISION_MARKER)
                p = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
            text.append(prompt_template.format(p))
            images += own_text_images
        text_inputs = processor(
            text=text,
            images=images or None,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        hidden_states = text_encoder(**text_inputs, output_hidden_states=True).hidden_states

        # Each layer is scaled to unit RMS per token in float32, then every bundle concatenates its three layers
        dtype = text_encoder.dtype
        normalized = {}
        for layer in sorted({layer for layers in hidden_states_layers for layer in layers}):
            states = hidden_states[layer].float()
            normalized[layer] = (states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(dtype)
        bundles = [torch.cat([normalized[layer] for layer in layers], dim=-1) for layers in hidden_states_layers]
        return torch.stack(bundles, dim=1), text_inputs.attention_mask.bool()

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: list[PIL.Image.Image] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        text_images: list[PIL.Image.Image] | list[list[PIL.Image.Image]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            prompt (`str` or `list[str]`):
                The prompt or prompts to encode. With `image`, each prompt is an edit instruction, or a JSON object
                with an `"edit_instruction"` key and the caption of the result: its other keys, or a
                `"structured_caption"` key holding the caption or a short description. A generation prompt is encoded
                as given: `__call__` first puts a JSON caption in the form Fibo 2.0 was trained with.
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
            text_images (`list[PIL.Image.Image]` or `list[list[PIL.Image.Image]]`, *optional*):
                Images of the caption's text, one per `text_render` entry that has `text`, in order, or a list of them
                per prompt. Each entry gets the vision marker of its image, and Qwen3-VL sees the images after any
                images to edit.

        Returns:
            The text embeddings, and their mask, or `None` when no prompt was padded.
        """
        device = device or self._execution_device
        if text_images is not None and all(isinstance(img, PIL.Image.Image) for img in text_images):
            # The images of a single prompt's text
            text_images = [text_images]
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                self.text_encoder,
                self.processor,
                prompt,
                image,
                text_images,
                device,
                max_sequence_length=self.tokenizer_max_length,
                prompt_template=self.prompt_template_encode,
                hidden_states_layers=self.text_encoder_out_layers,
            )

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

    def prepare_image_latents(self, image, batch_size, device, dtype):
        # The images to edit, as `image_processor.preprocess` prepared them, are normalized with the VAE's BatchNorm
        # statistics, like the image being generated. The pixels are made contiguous so the VAE's convolutions run the
        # same kernels as in the reference implementation
        image_latents = []
        for img in image:
            img = img.contiguous().to(device=device, dtype=self.vae.dtype)
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
        text_images=None,
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
                    edit = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if not isinstance(edit, dict):
                    continue
                if "edit_instruction" not in edit:
                    raise ValueError(
                        "To edit an image, `prompt` has to be the edit instruction, or a JSON caption of the result with"
                        f' an "edit_instruction" key, but is {p}'
                    )
                if "structured_caption" in edit and (
                    not isinstance(edit["structured_caption"], (dict, str)) or len(edit) > 2
                ):
                    raise ValueError(
                        'An edit prompt with a "structured_caption" key has to hold the caption of the result there, as'
                        f' a JSON object or a string, and nothing else but the "edit_instruction", but is {p}'
                    )

        if mask is not None:
            if image is None or len(image) != 1:
                raise ValueError("`mask` needs exactly one image in `image`.")
            if not isinstance(mask, PIL.Image.Image):
                raise ValueError(f"`mask` has to be a `PIL.Image.Image` but is {type(mask)}")
            if mask.size != image[0].size:
                raise ValueError(f"`mask` has size {mask.size} but `image` has size {image[0].size}.")

        if text_images is not None:
            if not isinstance(prompt, str):
                raise ValueError(f"`text_images` belong to a single caption, but `prompt` is {prompt}")
            if not all(isinstance(img, PIL.Image.Image) for img in text_images):
                raise ValueError("`text_images` has to be a list of `PIL.Image.Image`.")
            caption = self._parse_prompt(prompt)[0]
            if not isinstance(caption, dict):
                raise ValueError(f"`text_images` needs a JSON caption in `prompt`, but it is {prompt}")
            num_texts = sum(
                isinstance(entry, dict) and isinstance(entry.get("text"), str) and bool(entry["text"])
                for entry in caption.get("text_render") or []
            )
            if len(text_images) != num_texts:
                raise ValueError(
                    f"`text_images` holds {len(text_images)} images, but the caption has {num_texts} `text_render`"
                    " entries with text: pass one image per entry, in order."
                )

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
        text_images: list[PIL.Image.Image] | None = None,
        render_text_images: bool = True,
        fonts: list[str] | None = None,
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
                The prompt to guide the image generation, usually a structured JSON prompt, which is cleaned as Fibo
                2.0's training cleaned captions: empty values and the `aesthetic_score` and `preference_score` fields
                are dropped, and it is serialized compactly. With `image`, the edit instruction, or a JSON object with
                an `"edit_instruction"` key and the caption of the result: its other keys, or a `"structured_caption"`
                key holding the caption or a short description.
            image (`PIL.Image.Image` or `list[PIL.Image.Image]`, *optional*):
                The image to edit, or up to 5 images to edit together, which the instruction can refer to as
                `<image_1>`, `<image_2>` and so on. Without `image`, the pipeline generates an image from the prompt.
            mask (`PIL.Image.Image`, *optional*):
                The region of `image` to regenerate, white on black, when editing a single image. As in training, the
                region is greyed out before the image is encoded, through the mask made binary at 128.
            text_images (`list[PIL.Image.Image]`, *optional*):
                Images of the text the caption asks for, one per `text_render` entry that has `text`, in order: each
                entry's text drawn on its own, as `render_text_images` draws them. Qwen3-VL sees them with the prompt,
                each marked in its caption entry, which helps Fibo 2.0 render the text. Needs `prompt` as a single JSON
                caption, and takes precedence over `render_text_images`. A generation runs its first steps, while the
                noise level is above `text_images_max_noise_level` (0.9), without them; an edit uses them at every
                step.
            render_text_images (`bool`, *optional*, defaults to `True`):
                Whether to draw the `text_images` of each caption, the way Fibo 2.0's training drew them: each text in
                one line, black on white, in Pillow's built-in font at size 20, and the characters that font lacks in
                the first of `fonts` that has them. A caption without text, and a prompt without a JSON caption, such
                as a short description, get no images, as in training.
            fonts (`list[str]`, *optional*):
                Paths of the fonts to draw the characters of `text_images` that Pillow's built-in font lacks, in order.
                Training drew them in Aileron, then DejaVu Sans, which are in the `fonts` folder of
                `briaai/Fibo-2.0-dev`. Without them, such characters are drawn as Pillow's built-in font draws them.
            height (`int`, *optional*):
                The height in pixels of the generated image. Must be divisible by 16. Defaults to 1024, or when
                editing, to the size Fibo 2.0 was trained at for the aspect ratio of the first image, about one
                megapixel.
            width (`int`, *optional*):
                The width in pixels of the generated image. Must be divisible by 16. Defaults like `height`.
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Classifier-free guidance scale. Values above 1 run the negative prompt as a second pass at every step,
                and 1.0 turns guidance off.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt to guide away from when guidance is on. Defaults to the null caption Fibo 2.0 was trained
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
            text_images,
        )
        prompts = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None and image is None:
            # A generation caption in its trained form, serialized compactly. A prompt that isn't a JSON caption, such
            # as a short description, stays as it is. An edit prompt's caption takes it when the prompt is encoded
            prompts = [
                p if isinstance(caption, str) else json.dumps(caption or {}, separators=(",", ":"), ensure_ascii=False)
                for p, (caption, _) in zip(prompts, map(self._parse_prompt, prompts))
            ]
            prompt = prompts[0] if isinstance(prompt, str) else prompts
        # The images of each prompt's text: the caller's, or drawn from its caption's text_render entries that have
        # text, as Fibo 2.0's training drew them
        if text_images is not None:
            text_images = [list(text_images)]
        elif render_text_images and prompt is not None:
            fallback_fonts = [load_font(path) for path in fonts or []]
            text_images = []
            for caption, _ in map(self._parse_prompt, prompts):
                entries = (caption.get("text_render") or []) if isinstance(caption, dict) else []
                text_images.append(
                    [
                        draw_text_image(entry["text"], fallback_fonts)
                        for entry in entries
                        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]
                    ]
                )
        if not any(text_images or []):
            text_images = None
        # The size Fibo 2.0 was trained at for an aspect ratio, about the default area
        default_area = (self.default_sample_size * self.vae_scale_factor) ** 2
        if image is not None and height is None and width is None:
            width, height = self.image_processor.get_image_size(image[0].width / image[0].height, area=default_area)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._interrupt = False
        device = self._execution_device

        # 2. Prepare the images to edit, as Fibo 2.0 was trained on them: the crops the VAE encodes, and Qwen3-VL's copies
        vae_images = text_encoder_images = None
        if image is not None:
            vae_images, text_encoder_images = self.image_processor.preprocess(
                image,
                height=height,
                width=width,
                mask=mask,
                area=default_area,
                text_encoder_image_size=self.text_encoder_image_size,
            )

        # 3. Encode the prompt with the images to edit, and the negative prompt, without them, when guidance is on
        if prompt is not None:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            text_encoder_images,
            device,
            num_images_per_prompt,
            prompt_embeds,
            prompt_embeds_mask,
            text_images=text_images,
        )
        # A generation with images of its text runs its first steps on the captions without them: each text entry's
        # marker left empty
        base_prompt_embeds = base_prompt_embeds_mask = None
        if text_images is not None and image is None:
            base_prompts = [
                json.dumps(self._mark_text_images(json.loads(p), ""), separators=(",", ":"), ensure_ascii=False)
                if own
                else p
                for p, own in zip(prompts, text_images)
            ]
            base_prompt_embeds, base_prompt_embeds_mask = self.encode_prompt(
                base_prompts, device=device, num_images_per_prompt=num_images_per_prompt
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
                use_text_images = base_prompt_embeds is None or float(t) / 1000 <= self.text_images_max_noise_level
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds if use_text_images else base_prompt_embeds,
                    encoder_attention_mask=prompt_embeds_mask if use_text_images else base_prompt_embeds_mask,
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
