# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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

import torch

from ...models import ErnieImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ErnieImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _pad_text(
    text_hiddens: list[torch.Tensor], device: torch.device, dtype: torch.dtype, text_in_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length text hidden states to a common length and return (padded, lengths)."""
    batch_size = len(text_hiddens)
    if batch_size == 0:
        return (
            torch.zeros((0, 0, text_in_dim), device=device, dtype=dtype),
            torch.zeros((0,), device=device, dtype=torch.long),
        )
    normalized = [t.squeeze(1).to(device).to(dtype) if t.dim() == 3 else t.to(device).to(dtype) for t in text_hiddens]
    lengths = torch.tensor([t.shape[0] for t in normalized], device=device, dtype=torch.long)
    max_length = int(lengths.max().item())
    padded = torch.zeros((batch_size, max_length, text_in_dim), device=device, dtype=dtype)
    for i, t in enumerate(normalized):
        padded[i, : t.shape[0], :] = t
    return padded, lengths


class ErnieImageTextInputStep(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def description(self) -> str:
        return (
            "Input processing step that pads the variable-length text hidden states to a common length and "
            "produces `text_bth` / `text_lens` tensors consumed by the denoiser."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", ErnieImageTransformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=list,
                description="List of per-prompt text embeddings from the text encoder step.",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=list,
                description="List of per-prompt negative text embeddings from the text encoder step.",
            ),
            InputParam(
                "num_images_per_prompt",
                type_hint=int,
                default=1,
                description="Number of images to generate per prompt.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("batch_size", type_hint=int, description="The number of prompts in the batch."),
            OutputParam(
                "text_bth",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Padded text hidden states of shape (B, T_max, H) fed into the transformer.",
            ),
            OutputParam(
                "text_lens",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Actual per-prompt text lengths used to build the transformer attention mask.",
            ),
            OutputParam(
                "negative_text_bth",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Padded negative text hidden states, when classifier-free guidance is enabled.",
            ),
            OutputParam(
                "negative_text_lens",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Actual per-prompt negative text lengths, when classifier-free guidance is enabled.",
            ),
        ]

    @staticmethod
    def _expand(hiddens: list[torch.Tensor], num_images_per_prompt: int) -> list[torch.Tensor]:
        if num_images_per_prompt == 1:
            return list(hiddens)
        return [h for h in hiddens for _ in range(num_images_per_prompt)]

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype
        text_in_dim = components.text_in_dim
        num_images_per_prompt = block_state.num_images_per_prompt

        prompt_embeds = block_state.prompt_embeds
        block_state.batch_size = len(prompt_embeds)

        prompt_embeds = self._expand(prompt_embeds, num_images_per_prompt)
        text_bth, text_lens = _pad_text(prompt_embeds, device, dtype, text_in_dim)
        block_state.text_bth = text_bth
        block_state.text_lens = text_lens

        negative_prompt_embeds = block_state.negative_prompt_embeds
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = self._expand(negative_prompt_embeds, num_images_per_prompt)
            negative_text_bth, negative_text_lens = _pad_text(negative_prompt_embeds, device, dtype, text_in_dim)
            block_state.negative_text_bth = negative_text_bth
            block_state.negative_text_lens = negative_text_lens
        else:
            block_state.negative_text_bth = None
            block_state.negative_text_lens = None

        self.set_block_state(state, block_state)
        return components, state


class ErnieImageSetTimestepsStep(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def description(self) -> str:
        return "Step that sets the scheduler's timesteps for inference using a linear sigma schedule."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "num_inference_steps",
                type_hint=int,
                default=50,
                description="Number of denoising steps.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="The timesteps to use for inference."),
            OutputParam("num_inference_steps", type_hint=int, description="The number of denoising steps."),
        ]

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        num_inference_steps = block_state.num_inference_steps

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        components.scheduler.set_timesteps(sigmas=sigmas, device=device)

        block_state.timesteps = components.scheduler.timesteps
        block_state.num_inference_steps = num_inference_steps

        self.set_block_state(state, block_state)
        return components, state


class ErnieImagePrepareLatentsStep(ModularPipelineBlocks):
    model_name = "ernie-image"

    @property
    def description(self) -> str:
        return "Prepare random noise latents for the ErnieImage text-to-image denoising process."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", ErnieImageTransformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("height", type_hint=int, description="The height in pixels of the generated image."),
            InputParam("width", type_hint=int, description="The width in pixels of the generated image."),
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Pre-generated noisy latents. If provided, skips noise sampling.",
            ),
            InputParam(
                "generator",
                type_hint=torch.Generator,
                description="Torch generator for deterministic noise sampling.",
            ),
            InputParam(
                "text_bth",
                required=True,
                type_hint=torch.Tensor,
                description="Padded text hidden states; used to derive the total batch size for the latents.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="The initial noise latents to denoise."),
            OutputParam("height", type_hint=int, description="The resolved image height in pixels."),
            OutputParam("width", type_hint=int, description="The resolved image width in pixels."),
        ]

    @staticmethod
    def _check_inputs(components: ErnieImageModularPipeline, height: int, width: int) -> None:
        vae_scale_factor = components.vae_scale_factor
        if height % vae_scale_factor != 0 or width % vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {vae_scale_factor}, got {height} and {width}."
            )

    @torch.no_grad()
    def __call__(self, components: ErnieImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        self._check_inputs(components, height, width)

        total_batch_size = block_state.text_bth.shape[0]
        latent_h = height // components.vae_scale_factor
        latent_w = width // components.vae_scale_factor
        num_channels_latents = components.num_channels_latents

        shape = (total_batch_size, num_channels_latents, latent_h, latent_w)
        if block_state.latents is None:
            block_state.latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=dtype)
        else:
            block_state.latents = block_state.latents.to(device=device, dtype=dtype)

        block_state.height = height
        block_state.width = width

        self.set_block_state(state, block_state)
        return components, state
