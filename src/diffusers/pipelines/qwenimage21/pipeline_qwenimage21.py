# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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
import math
import random
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import QwenImageLoraLoaderMixin
from ...models import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
from ...models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from ..qwenimage.pipeline_output import QwenImagePipelineOutput


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
        >>> from diffusers import QwenImage21Pipeline

        >>> pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A capybara wearing a wizard hat, reading a book by candlelight, oil painting"
        >>> image = pipe(prompt, num_inference_steps=50).images[0]
        >>> image.save("qwenimage21.png")
        ```
"""


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
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


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.retrieve_timesteps
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


# Copied from diffusers.pipelines.flux.pipeline_flux_control_img2img.retrieve_latents
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


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.calculate_dimensions
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


class QwenImage21Pipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    r"""
    Text-to-image and image-conditioned generation with Qwen-Image 2.1.

    Prompt and condition images are encoded together by a Qwen3-VL model, so a condition image occupies the vision
    slots the encoder reserved for it and the transformer sees one interleaved text/image sequence.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler used to denoise the encoded image latents.
        vae ([`AutoencoderKLQwenImage21`]):
            Variational auto-encoder mapping images to and from the 64-channel latent space.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            Qwen3-VL model producing the joint text/image embeddings.
        processor ([`Qwen3VLProcessor`]):
            Processor that builds the chat template and tokenizes prompt and condition images.
        transformer ([`QwenImage21Transformer2DModel`]):
            The single-stream block-causal transformer that denoises the latents.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage21,
        text_encoder: Qwen3VLForConditionalGeneration,
        processor: Qwen3VLProcessor,
        transformer: QwenImage21Transformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        # The VAE compresses 16x spatially and the transformer consumes latents unpatched, so one token covers a 16x16
        # pixel tile.
        self.vae_scale_factor = 16
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 64
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.latent_channels
        )
        self.sys_prompt = "Comprehend and analyze the provided prompt."
        # The prompt is built as a raw template string and passed straight to
        # `self.processor(text=..., images=...)`, rather than going through `apply_chat_template`:
        # the two tokenize differently and the checkpoint expects this one. The "Picture 1: ..."
        # vision prefix only appears in the image-conditioned template.
        self.prompt_template_t2i = (
            f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{{}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prompt_template_ti2i = (
            f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.ref_token_list = ["Picture ", "Image ", "图 ", "图片 "]
        # Number of leading system-role tokens to drop from the hidden states. Derived from the
        # tokenized system message rather than hardcoded, so it tracks the processor's template.
        sys_message = [{"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]}]
        sys_tokens = self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
        self._drop_idx_t2i = len(sys_tokens[0])
        self._drop_idx_ti2i = self._drop_idx_t2i
        self._img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
        self._max_length = 8192

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    @staticmethod
    def _downsample_image_pad_tokens(hidden_states_list, image_pad_mask_list):
        """Collapse consecutive `<|image_pad|>` tokens into one per contiguous region.

        The vision-language processor expands each condition image into many vision tokens, but the transformer expects
        one token per image slot, which it then expands 4x. Keep the first token of each contiguous image-pad region
        and drop the rest.
        """
        out_hs, out_mask = [], []
        for hidden_state, pad_mask in zip(hidden_states_list, image_pad_mask_list):
            non_pad = ~pad_mask
            non_pad_tokens = hidden_state[non_pad]
            non_pad_mask = pad_mask[non_pad]

            pad_indices = torch.where(pad_mask)[0]
            insert_tokens, insert_positions = [], []
            if len(pad_indices) > 0:
                region_starts = [pad_indices[0].item()]
                if len(pad_indices) > 1:
                    diff = torch.diff(pad_indices)
                    for j, d in enumerate(diff):
                        if d > 1:
                            region_starts.append(pad_indices[j + 1].item())

                for start_idx in region_starts:
                    insert_pos = non_pad[:start_idx].sum().item()
                    insert_tokens.append(hidden_state[start_idx])
                    insert_positions.append(insert_pos)

            result_tokens = non_pad_tokens
            result_mask = non_pad_mask
            if insert_positions:
                for idx in sorted(range(len(insert_positions)), key=lambda i: insert_positions[i], reverse=True):
                    pos = insert_positions[idx]
                    result_tokens = torch.cat(
                        [result_tokens[:pos], insert_tokens[idx].unsqueeze(0), result_tokens[pos:]]
                    )
                    result_mask = torch.cat(
                        [
                            result_mask[:pos],
                            torch.tensor([True], dtype=torch.bool, device=result_mask.device),
                            result_mask[pos:],
                        ]
                    )

            out_hs.append(result_tokens)
            out_mask.append(result_mask)
        return out_hs, out_mask

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        image: list | None = None,
        device: torch.device | None = None,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        is_t2i = image is None

        if is_t2i:
            prompts = [self.prompt_template_t2i.format(t) for t in prompt]
            drop_idx = self._drop_idx_t2i
        else:
            prompts = []
            condition_pil_list = []
            for t in prompt:
                n_imgs = len(image)
                replace = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
                for i in range(2, n_imgs + 1):
                    replace += f" Picture {i}: <|vision_start|><|image_pad|><|vision_end|>"
                template = self.prompt_template_ti2i.replace(
                    "Picture 1: <|vision_start|><|image_pad|><|vision_end|>",
                    replace.replace("Picture ", random.choice(self.ref_token_list)),
                )
                prompts.append(template.format(t))
            for img in image:
                if not isinstance(img, PILImage.Image):
                    img = PILImage.fromarray(img)
                condition_pil_list.append(img)
            drop_idx = self._drop_idx_ti2i

        processor_kwargs = {"text": prompts, "padding": True, "return_tensors": "pt"}
        if not is_t2i:
            processor_kwargs["images"] = condition_pil_list

        model_inputs = self.processor(**processor_kwargs).to(device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
        }
        if not is_t2i and hasattr(model_inputs, "pixel_values"):
            forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        outputs = self.text_encoder(**forward_kwargs)
        hidden_states = outputs.hidden_states[-1]

        split_hidden_states = list(self._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        image_pad_mask = [
            (sample_ids[sample_mask.bool()] == self._img_token_id)
            for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
        ]
        image_pad_mask = [e[drop_idx:] for e in image_pad_mask]

        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])

        return prompt_embeds, encoder_attention_mask, image_pad_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: list[PipelineImageInput] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ):
        r"""
        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt to be encoded.
            image (`list[PipelineImageInput]`, *optional*):
                Condition images to encode alongside the prompt.
            device (`torch.device`):
                Torch device.
            num_images_per_prompt (`int`):
                Number of images generated per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Skips encoding when provided.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask, image_pad_mask = self._get_qwen_prompt_embeds(prompt, image, device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask, image_pad_mask

    def check_inputs(self, prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and "
                f"{width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Pass one of `prompt` or `prompt_embeds`.")

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        # 2.1 consumes latents unpatched, so packing is a plain spatial flatten.
        return latents.view(batch_size, num_channels_latents, height * width).transpose(1, 2)

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, _, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.transpose(1, 2).reshape(batch_size, channels, 1, height, width)
        return latents

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def prepare_latents(
        self, images, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        image_latents = None
        if images is not None:
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                encoded = image if image.shape[1] == self.latent_channels else self._encode_vae_image(image, generator)
                if batch_size > encoded.shape[0]:
                    if batch_size % encoded.shape[0] != 0:
                        raise ValueError(
                            f"Cannot duplicate `image` of batch size {encoded.shape[0]} to {batch_size} text prompts."
                        )
                    encoded = torch.cat([encoded] * (batch_size // encoded.shape[0]), dim=0)
                image_latent_height, image_latent_width = encoded.shape[3:]
                all_image_latents.append(
                    self._pack_latents(
                        encoded, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                )
            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            shape = (batch_size, 1, num_channels_latents, height, width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

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
        prompt: str | list[str] = None,
        image: PipelineImageInput | None = None,
        negative_prompt: str | list[str] = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        output_resolution: int = 1024,
        use_kv_cache: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt to guide image generation. Pass `prompt_embeds` instead to supply embeddings directly.
            image (`PipelineImageInput`, *optional*):
                One or more condition images. They are encoded by the text encoder as vision context and by the VAE
                into latent tokens prepended to the noise.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt not to guide image generation. Ignored when `true_cfg_scale` is not greater than 1.
            true_cfg_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Enabled by `true_cfg_scale > 1` together with a negative prompt.
            height (`int`, *optional*):
                Height in pixels of the generated image. Derived from the condition image's aspect ratio if omitted.
            width (`int`, *optional*):
                Width in pixels of the generated image. Derived from the condition image's aspect ratio if omitted.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps.
            sigmas (`list[float]`, *optional*):
                Custom sigmas for the denoising schedule.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images generated per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format, `"pil"`, `"np"`, `"pt"` or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Passed through to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                Called at the end of each denoising step.
            output_resolution (`int`, *optional*, defaults to 1024):
                Target side length used to derive `height`/`width` and to resize condition images.
            use_kv_cache (`bool`, *optional*, defaults to `True`):
                Cache the text and condition-image keys and values after the first step. Valid because
                `causal_condition` modulates those tokens from `t = 0`, making their activations step-independent.

                Toggling this does not reproduce the same image bit-for-bit in reduced precision. Caching makes the
                decode step attend with a different sequence layout than the prefill step, so the two tile differently
                and land on different rounding; both agree with an fp32 reference to the same tolerance. A one-ULP
                difference at the first block is then amplified by 32 blocks and every sampler step, so the two
                settings give equally valid but visibly distinct samples. Fix a sample by fixing this flag.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple` whose first
            element is a list with the generated images.
        """
        if image is not None:
            image_size = image[-1].size if isinstance(image, list) else image.size
            calculated_width, calculated_height, _ = calculate_dimensions(
                output_resolution * output_resolution, image_size[0] / image_size[1]
            )
            height = height or calculated_height
            width = width or calculated_width
        height = height or output_resolution
        width = width or output_resolution

        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 1. Preprocess condition images: one resize feeds both the text encoder and the VAE.
        input_image_sizes, input_images, vae_images = [], None, None
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = image if isinstance(image, list) else [image]
            input_images, vae_images = [], []
            for img in image:
                if hasattr(img, "mode") and img.mode != "RGBA":
                    img = img.convert("RGBA")
                image_width, image_height = img.size
                input_width, input_height, _ = calculate_dimensions(
                    output_resolution * output_resolution, image_width / image_height
                )
                input_image_sizes.append((input_width, input_height))
                input_images.append(self.image_processor.resize(img, width=input_width, height=input_height))
                vae_images.append(
                    self.image_processor.preprocess(img, width=input_width, height=input_height).unsqueeze(2)
                )

        # 2. Encode prompt
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no "
                f"negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                "negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        prompt_embeds, prompt_embeds_mask, image_pad_mask = self.encode_prompt(
            image=input_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask, negative_image_pad_mask = self.encode_prompt(
                image=input_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        # 3. Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents, input_images_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        img_shapes = [
            [
                *[
                    (1, vae_height // self.vae_scale_factor, vae_width // self.vae_scale_factor)
                    for vae_width, vae_height in input_image_sizes
                ],
                (1, height // self.vae_scale_factor, width // self.vae_scale_factor),
            ]
        ] * batch_size

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            latents.shape[1],
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # The transformer's `img_mask` spans the joint sequence, so append one slot per 2x2 group of target latents.
        target_slots = torch.ones(
            [latents.shape[0], latents.shape[1] // 4], dtype=image_pad_mask.dtype, device=image_pad_mask.device
        )
        image_pad_mask = torch.cat([image_pad_mask, target_slots], dim=1)
        if do_true_cfg:
            negative_image_pad_mask = torch.cat([negative_image_pad_mask, target_slots], dim=1)

        # Text and condition-image keys and values are step-independent under `causal_condition`, so the first step
        # prefills them and later steps only recompute the target image's tokens.
        num_blocks = len(self.transformer.transformer_blocks)
        cache_enabled = use_kv_cache and self.transformer.config.causal_condition
        cond_cache = QwenImage21KVCache(num_blocks) if cache_enabled else None
        neg_cache = QwenImage21KVCache(num_blocks) if cache_enabled and do_true_cfg else None

        # 5. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                kv_mode = "extract" if (cache_enabled and i == 0) else ("cached" if cache_enabled else None)

                latent_model_input = latents
                if input_images_latents is not None:
                    latent_model_input = torch.cat([input_images_latents, latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        img_shapes=img_shapes,
                        img_mask=image_pad_mask,
                        attention_kwargs=self.attention_kwargs,
                        kv_cache=cond_cache,
                        kv_cache_mode=kv_mode,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred[:, -latents.size(1) :]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            img_shapes=img_shapes,
                            img_mask=negative_image_pad_mask,
                            attention_kwargs=self.attention_kwargs,
                            kv_cache=neg_cache,
                            kv_cache_mode=kv_mode,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, -latents.size(1) :]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug:
                    # https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
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
            latents = latents * latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
