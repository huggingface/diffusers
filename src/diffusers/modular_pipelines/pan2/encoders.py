# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import PIL.Image
import torch
from transformers import Qwen2Tokenizer, Qwen3_5TextModel

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLWan
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import PAN2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_pan2_prompt_embeds(
    text_encoder: Qwen3_5TextModel,
    tokenizer: Qwen2Tokenizer,
    prompt: str,
    system_message: str,
    num_hidden_layers_to_skip: int,
    max_sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    # An empty prompt is encoded as a single space, as in training.
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt or " "}]

    # PAN2 conditions on the tokens after the `<|im_start|>user\n` header: the prompt and the assistant header after it.
    header_ids = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True
    ).input_ids
    crop_start = next(
        i + len(header_ids) for i in range(len(input_ids)) if input_ids[i : i + len(header_ids)] == header_ids
    )
    input_ids = torch.tensor([input_ids[: crop_start + max_sequence_length]], device=device)

    prompt_embeds = text_encoder(input_ids=input_ids, output_hidden_states=True).hidden_states[
        -(num_hidden_layers_to_skip + 1)
    ]
    return prompt_embeds[:, crop_start:]


class PAN2TextEncoderStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that encodes the prompt, and the negative prompt unless guidance is disabled, with "
            "Qwen3.5."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3_5TextModel),
            ComponentSpec("tokenizer", Qwen2Tokenizer),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [ConfigSpec(name="enable_safety_checker", default=True)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
            InputParam.template(
                "negative_prompt", description="The prompt not to guide generation, a single string.", default=""
            ),
            InputParam.template("max_sequence_length", default=1536),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
        ]

    @staticmethod
    def check_inputs(prompt, negative_prompt):
        # Every prompt is its own attention sequence without padding, so a call takes a single prompt.
        if not isinstance(prompt, str):
            raise ValueError(f"`prompt` must be a single string, got {type(prompt)}.")
        if not isinstance(negative_prompt, str):
            raise ValueError(f"`negative_prompt` must be a single string, got {type(negative_prompt)}.")

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.prompt, block_state.negative_prompt)
        device = components._execution_device

        if components.requires_safety_checker:
            components.load_safety_checker()
            components.safety_checker.to(device)
            try:
                if not components.safety_checker.check_text_safety(block_state.prompt):
                    raise ValueError("The PAN2 guardrail detected unsafe content in the prompt.")
            finally:
                components.safety_checker.to("cpu")

        block_state.prompt_embeds = get_pan2_prompt_embeds(
            components.text_encoder,
            components.tokenizer,
            block_state.prompt,
            system_message=components.system_message,
            num_hidden_layers_to_skip=components.text_encoder_num_hidden_layers_to_skip,
            max_sequence_length=block_state.max_sequence_length,
            device=device,
        )

        # The guider belongs to the denoise step: in the full pipeline, it decides whether the negative prompt is
        # needed. Standalone, there is no guider and the negative prompt is encoded, since the PAN2 denoise step uses
        # classifier-free guidance by default.
        block_state.negative_prompt_embeds = None
        if getattr(components, "guider", None) is None or components.requires_unconditional_embeds:
            block_state.negative_prompt_embeds = get_pan2_prompt_embeds(
                components.text_encoder,
                components.tokenizer,
                block_state.negative_prompt,
                system_message=components.system_message,
                num_hidden_layers_to_skip=components.text_encoder_num_hidden_layers_to_skip,
                max_sequence_length=block_state.max_sequence_length,
                device=device,
            )
        elif block_state.negative_prompt:
            logger.warning("`negative_prompt` is ignored because the guider does not use unconditional embeddings.")

        self.set_block_state(state, block_state)
        return components, state


def resize_and_center_crop(image: PIL.Image.Image, height: int, width: int) -> PIL.Image.Image:
    r"""Scale the image to cover the `(height, width)` canvas, rounding its size, and center crop it, as PAN2 was trained."""
    scale = max(width / image.width, height / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), resample=PIL.Image.LANCZOS)
    left, top = round((image.width - width) / 2), round((image.height - height) / 2)
    return image.crop((left, top, left + width, top + height))


class PAN2VaeEncoderStep(ModularPipelineBlocks):
    model_name = "pan2"

    @property
    def description(self) -> str:
        return (
            "VAE encoder step that resizes and center-crops the first frame onto the target canvas and encodes it into "
            "normalized image latents."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "image", required=True, type_hint=PIL.Image.Image, description="The first frame, a single image."
            ),
            InputParam.template("height", description="The height in pixels of the generated video, 704 by default."),
            InputParam.template("width", description="The width in pixels of the generated video, 1248 by default."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="The normalized latents of the first frame, of shape `(1, 48, 1, h, w)`.",
            ),
        ]

    @staticmethod
    def check_inputs(image):
        # A call takes a single prompt, so it takes a single first frame too.
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` must be a single PIL image, got {type(image)}.")

    @torch.no_grad()
    def __call__(
        self, components: PAN2ModularPipeline, state: PipelineState
    ) -> tuple[PAN2ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.image)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        device = components._execution_device
        vae = components.vae

        image = resize_and_center_crop(block_state.image, block_state.height, block_state.width)
        image = components.video_processor.preprocess(image, height=block_state.height, width=block_state.width)
        image = image.unsqueeze(2).to(device=device, dtype=vae.dtype)
        image_latents = vae.encode(image).latent_dist.mode()

        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(image_latents)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(image_latents)
        block_state.image_latents = (image_latents - latents_mean) / latents_std

        self.set_block_state(state, block_state)
        return components, state
