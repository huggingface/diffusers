# Copyright 2025 VisualCloze team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from ...loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..flux.pipeline_flux_fill import calculate_shift, retrieve_latents, retrieve_timesteps
from ..flux.pipeline_output import FluxPipelineOutput
from ..pipeline_utils import DiffusionPipeline
from .visualcloze_utils import VisualClozeProcessor


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
        >>> from diffusers import VisualClozeGenerationPipeline, FluxFillPipeline as VisualClozeUpsamplingPipeline
        >>> from diffusers.utils import load_image
        >>> from PIL import Image

        >>> image_paths = [
        ...     # in-context examples
        ...     [
        ...         load_image(
        ...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg"
        ...         ),
        ...         load_image(
        ...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg"
        ...         ),
        ...     ],
        ...     # query with the target image
        ...     [
        ...         load_image(
        ...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg"
        ...         ),
        ...         None,  # No image needed for the target image
        ...     ],
        ... ]
        >>> task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
        >>> content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."
        >>> pipe = VisualClozeGenerationPipeline.from_pretrained(
        ...     "VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     task_prompt=task_prompt,
        ...     content_prompt=content_prompt,
        ...     image=image_paths,
        ...     guidance_scale=30,
        ...     num_inference_steps=30,
        ...     max_sequence_length=512,
        ...     generator=torch.Generator("cpu").manual_seed(0),
        ... ).images[0][0]

        >>> # optional, upsampling the generated image
        >>> pipe_upsample = VisualClozeUpsamplingPipeline.from_pipe(pipe)
        >>> pipe_upsample.to("cuda")

        >>> mask_image = Image.new("RGB", image.size, (255, 255, 255))

        >>> image = pipe_upsample(
        ...     image=image,
        ...     mask_image=mask_image,
        ...     prompt=content_prompt,
        ...     width=1344,
        ...     height=768,
        ...     strength=0.4,
        ...     guidance_scale=30,
        ...     num_inference_steps=30,
        ...     max_sequence_length=512,
        ...     generator=torch.Generator("cpu").manual_seed(0),
        ... ).images[0]

        >>> image.save("visualcloze.png")
        ```
"""


class VisualClozeGenerationPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The VisualCloze pipeline for image generation with visual context. Reference:
    https://github.com/lzyhha/VisualCloze/tree/main This pipeline is designed to generate images based on visual
    in-context examples.

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
        resolution (`int`, *optional*, defaults to 384):
            The resolution of each image when concatenating images from the query and in-context examples.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        resolution: int = 384,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.resolution = resolution
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VisualClozeProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, vae_latent_channels=self.latent_channels, resolution=resolution
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Modified from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
    def encode_prompt(
        self,
        layout_prompt: Union[str, List[str]],
        task_prompt: Union[str, List[str]],
        content_prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            layout_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to define the number of in-context examples and the number of images involved in
                the task.
            task_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to define the task intention.
            content_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to define the content or caption of the target image to be generated.
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        if isinstance(layout_prompt, str):
            layout_prompt = [layout_prompt]
            task_prompt = [task_prompt]
            content_prompt = [content_prompt]

        def _preprocess(prompt, content=False):
            if prompt is not None:
                return f"The last image of the last row depicts: {prompt}" if content else prompt
            else:
                return ""

        prompt = [
            f"{_preprocess(layout_prompt[i])} {_preprocess(task_prompt[i])} {_preprocess(content_prompt[i], content=True)}".strip()
            for i in range(len(layout_prompt))
        ]
        pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def check_inputs(
        self,
        image,
        task_prompt,
        content_prompt,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # Validate prompt inputs
        if (task_prompt is not None or content_prompt is not None) and prompt_embeds is not None:
            raise ValueError("Cannot provide both text `task_prompt` + `content_prompt` and `prompt_embeds`. ")

        if task_prompt is None and content_prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `task_prompt` + `content_prompt` or pre-computed `prompt_embeds`. ")

        # Validate prompt types and consistency
        if task_prompt is None:
            raise ValueError("`task_prompt` is missing.")

        if task_prompt is not None and not isinstance(task_prompt, (str, list)):
            raise ValueError(f"`task_prompt` must be str or list, got {type(task_prompt)}")

        if content_prompt is not None and not isinstance(content_prompt, (str, list)):
            raise ValueError(f"`content_prompt` must be str or list, got {type(content_prompt)}")

        if isinstance(task_prompt, list) or isinstance(content_prompt, list):
            if not isinstance(task_prompt, list) or not isinstance(content_prompt, list):
                raise ValueError(
                    f"`task_prompt` and `content_prompt` must both be lists, or both be of type str or None, "
                    f"got {type(task_prompt)} and {type(content_prompt)}"
                )
            if len(content_prompt) != len(task_prompt):
                raise ValueError("`task_prompt` and `content_prompt` must have the same length whe they are lists.")

            for sample in image:
                if not isinstance(sample, list) or not isinstance(sample[0], list):
                    raise ValueError("Each sample in the batch must have a 2D list of images.")
                if len({len(row) for row in sample}) != 1:
                    raise ValueError("Each in-context example and query should contain the same number of images.")
                if not any(img is None for img in sample[-1]):
                    raise ValueError("There are no targets in the query, which should be represented as None.")
                for row in sample[:-1]:
                    if any(img is None for img in row):
                        raise ValueError("Images are missing in in-context examples.")

        # Validate embeddings
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        # Validate sequence length
        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"max_sequence_length cannot exceed 512, got {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(image, vae_scale_factor, device, dtype):
        latent_image_ids = []

        for idx, img in enumerate(image, start=1):
            img = img.squeeze(0)
            channels, height, width = img.shape

            num_patches_h = height // vae_scale_factor // 2
            num_patches_w = width // vae_scale_factor // 2

            patch_ids = torch.zeros(num_patches_h, num_patches_w, 3, device=device, dtype=dtype)
            patch_ids[..., 0] = idx
            patch_ids[..., 1] = torch.arange(num_patches_h, device=device, dtype=dtype)[:, None]
            patch_ids[..., 2] = torch.arange(num_patches_w, device=device, dtype=dtype)[None, :]

            patch_ids = patch_ids.reshape(-1, 3)
            latent_image_ids.append(patch_ids)

        return torch.cat(latent_image_ids, dim=0)

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, sizes, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        start = 0
        unpacked_latents = []
        for i in range(len(sizes)):
            cur_size = sizes[i]
            height = cur_size[0][0] // vae_scale_factor
            width = sum([size[1] for size in cur_size]) // vae_scale_factor

            end = start + (height * width) // 4

            cur_latents = latents[:, start:end]
            cur_latents = cur_latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
            cur_latents = cur_latents.permute(0, 3, 1, 4, 2, 5)
            cur_latents = cur_latents.reshape(batch_size, channels // (2 * 2), height, width)

            unpacked_latents.append(cur_latents)

            start = end

        return unpacked_latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        depr_message = f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        depr_message = f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

    def _prepare_latents(self, image, mask, gen, vae_scale_factor, device, dtype):
        """Helper function to prepare latents for a single batch."""
        # Concatenate images and masks along width dimension
        image = [torch.cat(img, dim=3).to(device=device, dtype=dtype) for img in image]
        mask = [torch.cat(m, dim=3).to(device=device, dtype=dtype) for m in mask]

        # Generate latent image IDs
        latent_image_ids = self._prepare_latent_image_ids(image, vae_scale_factor, device, dtype)

        # For initial encoding, use actual images
        image_latent = [self._encode_vae_image(img, gen) for img in image]
        masked_image_latent = [img.clone() for img in image_latent]

        for i in range(len(image_latent)):
            # Rearrange latents and masks for patch processing
            num_channels_latents, height, width = image_latent[i].shape[1:]
            image_latent[i] = self._pack_latents(image_latent[i], 1, num_channels_latents, height, width)
            masked_image_latent[i] = self._pack_latents(masked_image_latent[i], 1, num_channels_latents, height, width)

            # Rearrange masks for patch processing
            num_channels_latents, height, width = mask[i].shape[1:]
            mask[i] = mask[i].view(
                1,
                num_channels_latents,
                height // vae_scale_factor,
                vae_scale_factor,
                width // vae_scale_factor,
                vae_scale_factor,
            )
            mask[i] = mask[i].permute(0, 1, 3, 5, 2, 4)
            mask[i] = mask[i].reshape(
                1,
                num_channels_latents * (vae_scale_factor**2),
                height // vae_scale_factor,
                width // vae_scale_factor,
            )
            mask[i] = self._pack_latents(
                mask[i],
                1,
                num_channels_latents * (vae_scale_factor**2),
                height // vae_scale_factor,
                width // vae_scale_factor,
            )

        # Concatenate along batch dimension
        image_latent = torch.cat(image_latent, dim=1)
        masked_image_latent = torch.cat(masked_image_latent, dim=1)
        mask = torch.cat(mask, dim=1)

        return image_latent, masked_image_latent, mask, latent_image_ids

    def prepare_latents(
        self,
        input_image,
        input_mask,
        timestep,
        batch_size,
        dtype,
        device,
        generator,
        vae_scale_factor,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Process each batch
        masked_image_latents = []
        image_latents = []
        masks = []
        latent_image_ids = []

        for i in range(len(input_image)):
            _image_latent, _masked_image_latent, _mask, _latent_image_ids = self._prepare_latents(
                input_image[i],
                input_mask[i],
                generator if isinstance(generator, torch.Generator) else generator[i],
                vae_scale_factor,
                device,
                dtype,
            )
            masked_image_latents.append(_masked_image_latent)
            image_latents.append(_image_latent)
            masks.append(_mask)
            latent_image_ids.append(_latent_image_ids)

        # Concatenate all batches
        masked_image_latents = torch.cat(masked_image_latents, dim=0)
        image_latents = torch.cat(image_latents, dim=0)
        masks = torch.cat(masks, dim=0)

        # Handle batch size expansion
        if batch_size > masked_image_latents.shape[0]:
            if batch_size % masked_image_latents.shape[0] == 0:
                # Expand batches by repeating
                additional_image_per_prompt = batch_size // masked_image_latents.shape[0]
                masked_image_latents = torch.cat([masked_image_latents] * additional_image_per_prompt, dim=0)
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
                masks = torch.cat([masks] * additional_image_per_prompt, dim=0)
            else:
                raise ValueError(
                    f"Cannot expand batch size from {masked_image_latents.shape[0]} to {batch_size}. "
                    "Batch sizes must be multiples of each other."
                )

        # Add noise to latents
        noises = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.scale_noise(image_latents, timestep, noises).to(dtype=dtype)

        # Combine masked latents with masks
        masked_image_latents = torch.cat((masked_image_latents, masks), dim=-1).to(dtype=dtype)

        return latents, masked_image_latents, latent_image_ids[0]

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

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
        task_prompt: Union[str, List[str]] = None,
        content_prompt: Union[str, List[str]] = None,
        image: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 30.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the VisualCloze pipeline for generation.

        Args:
            task_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to define the task intention.
            content_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to define the content or caption of the target image to be generated.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 30.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
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
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
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
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image,
            task_prompt,
            content_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        processor_output = self.image_processor.preprocess(
            task_prompt, content_prompt, image, vae_scale_factor=self.vae_scale_factor
        )

        # 2. Define call parameters
        if processor_output["task_prompt"] is not None and isinstance(processor_output["task_prompt"], str):
            batch_size = 1
        elif processor_output["task_prompt"] is not None and isinstance(processor_output["task_prompt"], list):
            batch_size = len(processor_output["task_prompt"])

        device = self._execution_device

        # 3. Prepare prompt embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            layout_prompt=processor_output["layout_prompt"],
            task_prompt=processor_output["task_prompt"],
            content_prompt=processor_output["content_prompt"],
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare timesteps
        # Calculate sequence length and shift factor
        image_seq_len = sum(
            (size[0] // self.vae_scale_factor // 2) * (size[1] // self.vae_scale_factor // 2)
            for sample in processor_output["image_size"][0]
            for size in sample
        )

        # Calculate noise schedule parameters
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        # Get timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, 1.0, device)

        # 5. Prepare latent variables
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents, masked_image_latents, latent_image_ids = self.prepare_latents(
            processor_output["init_image"],
            processor_output["mask"],
            latent_timestep,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            vae_scale_factor=self.vae_scale_factor,
        )

        # Calculate warmup steps
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Prepare guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = torch.cat((latents, masked_image_latents), dim=2)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # Some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # XLA optimization
                if XLA_AVAILABLE:
                    xm.mark_step()

        # 7. Post-process the image
        # Crop the target image
        # Since the generated image is a concatenation of the conditional and target regions,
        # we need to extract only the target regions based on their positions
        image = []
        if output_type == "latent":
            image = latents
        else:
            for b in range(len(latents)):
                cur_image_size = processor_output["image_size"][b % batch_size]
                cur_target_position = processor_output["target_position"][b % batch_size]
                cur_latent = self._unpack_latents(latents[b].unsqueeze(0), cur_image_size, self.vae_scale_factor)[-1]
                cur_latent = (cur_latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                cur_image = self.vae.decode(cur_latent, return_dict=False)[0]
                cur_image = self.image_processor.postprocess(cur_image, output_type=output_type)[0]

                start = 0
                cropped = []
                for i, size in enumerate(cur_image_size[-1]):
                    if cur_target_position[i]:
                        if output_type == "pil":
                            cropped.append(cur_image.crop((start, 0, start + size[1], size[0])))
                        else:
                            cropped.append(cur_image[0 : size[0], start : start + size[1]])
                    start += size[1]
                image.append(cropped)
            if output_type != "pil":
                image = np.concatenate([arr[None] for sub_image in image for arr in sub_image], axis=0)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
