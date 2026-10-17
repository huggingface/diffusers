# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import math

import PIL.Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLFlux2
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .image_processor import Flux3ImageProcessor
from .modular_pipeline import Flux3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Flux3ProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def description(self) -> str:
        return (
            "Image preprocess step for FLUX.3 image-to-image generation. Each reference image keeps its own "
            "resolution: it is converted to RGB, downscaled only when it exceeds 4096 x 4096 pixels, and "
            "center-cropped to a multiple of 16 pixels."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                Flux3ImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "vae_latent_channels": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "image",
                description="The reference image or images to condition the generation on; each keeps its own "
                "resolution (center-cropped to the 16-pixel grid)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_images",
                type_hint=list[torch.Tensor],
                description="The preprocessed reference images, one tensor of shape (1, 3, height_i, width_i) in [-1, 1] per image",
            )
        ]

    @torch.no_grad()
    def __call__(
        self, components: Flux3ModularPipeline, state: PipelineState
    ) -> tuple[Flux3ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        images = block_state.image if isinstance(block_state.image, list) else [block_state.image]
        condition_images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                raise ValueError(f"`image` must be a `PIL.Image.Image` or a list of them, got {type(image)}")
            # references keep their own resolution: downscaled only above 4096 x 4096 pixels, then center-cropped to
            # the 16-pixel grid (`resize_mode="crop"` is a pure crop for this processor)
            image = components.image_processor._resize_if_exceeds_area(image, 4096 * 4096)
            condition_images.append(components.image_processor.preprocess(image, resize_mode="crop"))
        block_state.condition_images = condition_images

        self.set_block_state(state, block_state)
        return components, state


class Flux3VaeEncoderStep(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def description(self) -> str:
        return (
            "VAE Encoder step that encodes the preprocessed reference images into normalized packed latents, one "
            "latent tensor per image."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLFlux2)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "condition_images",
                required=True,
                type_hint=list[torch.Tensor],
                description="The preprocessed reference images, one tensor of shape (1, 3, height_i, width_i) in [-1, 1] per image",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_cond_latents",
                type_hint=list[torch.Tensor],
                description="Normalized packed latents of the reference images, one tensor of shape (1, 128, height_i//16, width_i//16) per image",
            )
        ]

    @staticmethod
    def _patchify_latents(latents):
        """Convert latents to patchified format: (B, C, H, W) -> (B, C*4, H//2, W//2)."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        vae = components.vae
        device = components._execution_device

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, vae.dtype)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            device, vae.dtype
        )

        image_cond_latents = []
        for image in block_state.condition_images:
            latents = vae.encode(image.to(device=device, dtype=vae.dtype)).latent_dist.mode()
            latents = self._patchify_latents(latents)
            image_cond_latents.append((latents - latents_bn_mean) / latents_bn_std)
        block_state.image_cond_latents = image_cond_latents

        self.set_block_state(state, block_state)
        return components, state


class Flux3TextEncoderStep(ModularPipelineBlocks):
    model_name = "flux3"

    def __init__(self, pad_multiple: int = 80):
        # the text window is right-padded to a multiple of this many tokens (capped at `max_sequence_length`); the
        # padding is masked inside Qwen3-VL but not in the transformer, so the window size is part of the model recipe
        self.pad_multiple = pad_multiple
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Text Encoder step that generates text embeddings using Qwen3-VL to guide the generation. Takes one "
            f"prompt per call; it goes through the chat template and is right-padded to a multiple of "
            f"{self.pad_multiple} tokens."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("tokenizer", AutoProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide the generation (one prompt per call)"),
            # no `default=""`: a materialized default would be indistinguishable from a user-passed value,
            # and the guider-disabled warning depends on that distinction (the block falls back to "" itself)
            InputParam.template("negative_prompt", description="The prompt not to guide the generation"),
            InputParam.template(
                "max_sequence_length",
                default=8192,
                description="Maximum number of prompt tokens; longer prompts are truncated.",
            ),
            InputParam(
                "text_encoder_out_layers",
                type_hint=tuple[int],
                default=(4, 8, 12, 16, 20, 24, 28, 32),
                required=False,
                description="Indices of the text-encoder hidden layers whose states are stacked to form the text embeddings",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Text embeddings (stacked Qwen3-VL hidden states) used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Negative text embeddings used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        # one prompt per call (the text window is sized per prompt, and the transformer attends to its padding);
        # `num_images_per_prompt` generates several images from it
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                f"`prompt` must be a str; batched prompts are not supported, got {type(block_state.prompt).__name__}."
            )
        if block_state.negative_prompt is not None and not isinstance(block_state.negative_prompt, str):
            raise ValueError(
                "`negative_prompt` must be a str or None; batched prompts are not supported, "
                f"got {type(block_state.negative_prompt).__name__}."
            )

    @staticmethod
    def _get_qwen3_vl_prompt_embeds(
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: AutoProcessor,
        prompt: str,
        device: torch.device | None = None,
        max_sequence_length: int = 8192,
        pad_multiple: int = 80,
        hidden_states_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32),
    ):
        dtype = text_encoder.dtype
        device = text_encoder.device if device is None else device

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        # the window is the prompt's length rounded up to the pad multiple, capped at `max_sequence_length`
        real_length = tokenizer.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_sequence_length)[
            "input_ids"
        ].shape[1]
        padded_length = min(math.ceil(real_length / pad_multiple) * pad_multiple, max_sequence_length)
        inputs = tokenizer.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=padded_length,
        )

        # Qwen3-VL computes the position id of the padding tokens differently across transformers versions (1 in
        # 4.57, the training environment; 0 in 5.x). FLUX.3 attends to the padding tokens, so the position ids are
        # passed explicitly to match the training environment.
        input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
        position_ids = torch.arange(padded_length, device=device)[None].masked_fill(attention_mask == 0, 1)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them along the channel dimension
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        device = components._execution_device

        block_state.prompt_embeds = self._get_qwen3_vl_prompt_embeds(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            prompt=block_state.prompt,
            device=device,
            max_sequence_length=block_state.max_sequence_length,
            pad_multiple=self.pad_multiple,
            hidden_states_layers=block_state.text_encoder_out_layers,
        )

        # The guider is not a component of this block. Combined with the denoiser, the pipeline's guider decides
        # through `requires_unconditional_embeds` whether unconditional embeddings are needed (the negative prompt
        # then defaults to ""); with the guider disabled a provided `negative_prompt` is ignored with a warning.
        # Standalone (no guider), the negative prompt is encoded only when the caller passes one.
        has_guider = getattr(components, "guider", None) is not None
        if components.requires_unconditional_embeds or (block_state.negative_prompt is not None and not has_guider):
            block_state.negative_prompt_embeds = self._get_qwen3_vl_prompt_embeds(
                text_encoder=components.text_encoder,
                tokenizer=components.tokenizer,
                prompt=block_state.negative_prompt or "",
                device=device,
                max_sequence_length=block_state.max_sequence_length,
                pad_multiple=self.pad_multiple,
                hidden_states_layers=block_state.text_encoder_out_layers,
            )
        else:
            if block_state.negative_prompt is not None:
                logger.warning("`negative_prompt` is provided but the guider is disabled, it will be ignored.")
            block_state.negative_prompt_embeds = None

        self.set_block_state(state, block_state)
        return components, state


class Flux3DistilledTextEncoderStep(Flux3TextEncoderStep):
    model_name = "flux3-distilled"

    @property
    def description(self) -> str:
        return (
            "Text Encoder step that generates text embeddings using Qwen3-VL to guide the generation with the "
            "step-distilled FLUX.3 model, which samples without classifier-free guidance (no negative prompt). Takes "
            f"one prompt per call; it goes through the chat template and is right-padded to a multiple of "
            f"{self.pad_multiple} tokens."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide the generation (one prompt per call)"),
            InputParam.template(
                "max_sequence_length",
                default=8192,
                description="Maximum number of prompt tokens; longer prompts are truncated.",
            ),
            InputParam(
                "text_encoder_out_layers",
                type_hint=tuple[int],
                default=(4, 8, 12, 16, 20, 24, 28, 32),
                required=False,
                description="Indices of the text-encoder hidden layers whose states are stacked to form the text embeddings",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="Text embeddings (stacked Qwen3-VL hidden states) used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                f"`prompt` must be a str; batched prompts are not supported, got {type(block_state.prompt).__name__}."
            )

    @torch.no_grad()
    def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.prompt_embeds = self._get_qwen3_vl_prompt_embeds(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            prompt=block_state.prompt,
            device=components._execution_device,
            max_sequence_length=block_state.max_sequence_length,
            pad_multiple=self.pad_multiple,
            hidden_states_layers=block_state.text_encoder_out_layers,
        )

        self.set_block_state(state, block_state)
        return components, state
