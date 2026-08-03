# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLModel

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import InpaintProcessor, VaeImageProcessor
from ...models import AutoencoderKLQwenImage
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Indices into the Qwen3-VL `hidden_states` tuple (0 is the embedding output) whose states are stacked per token as the
# transformer's text conditioning. Must have `transformer.config.num_text_layers` entries.
KREA2_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Krea 2 wraps the prompt in this Qwen-Image chat template before encoding. The prompt is padded to a fixed length
# first and the assistant suffix is appended *after* the padding (matching how the model was sampled at training time);
# the first `_PROMPT_TEMPLATE_ENCODE_START_IDX` (system prefix) tokens are dropped from the encoder outputs.
_PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
_PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PROMPT_TEMPLATE_ENCODE_START_IDX = 34
_PROMPT_TEMPLATE_ENCODE_NUM_SUFFIX_TOKENS = 5

_REFERENCE_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
    "{}{}<|im_end|>\n<|im_start|>assistant\n"
)


class Krea2ReferenceImageProcessor(Qwen2VLImageProcessor):
    def __init__(
        self,
        size: dict | None = None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        super().__init__(
            size=size or {"longest_edge": 16777216, "shortest_edge": 65536},
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            image_mean=image_mean,
            image_std=image_std,
        )

    @property
    def device(self):
        if self._processor_device is None:
            raise AttributeError("Krea2ReferenceImageProcessor is device-independent")
        return self._processor_device

    @device.setter
    def device(self, value):
        self._processor_device = value


# auto_docstring
class Krea2TextEncoderStep(ModularPipelineBlocks):
    """
    Text encoder step that tokenizes the prompt(s) with the Krea 2 chat template, runs the Qwen3-VL text encoder, and
    stacks a fixed set of decoder-layer hidden states per token as the transformer's text conditioning. The negative
    prompt is encoded the same way when the guider enables CFG.

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. tokenizer (`AutoTokenizer`): The tokenizer paired
          with the text encoder. guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The negative prompt(s) for CFG.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`):
              Per-prompt negative text features (only when guidance is enabled).
          negative_prompt_embeds_mask (`Tensor`):
              Per-prompt negative text mask (only when guidance is enabled).
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that tokenizes the prompt(s) with the Krea 2 chat template, runs the Qwen3-VL text "
            "encoder, and stacks a fixed set of decoder-layer hidden states per token as the transformer's text "
            "conditioning. The negative prompt is encoded the same way when the guider enables CFG."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLModel, description="The Qwen3-VL text encoder."),
            ComponentSpec("tokenizer", AutoTokenizer, description="The tokenizer paired with the text encoder."),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.5, "use_original_formulation": True}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam(name="negative_prompt", type_hint=str, description="The negative prompt(s) for CFG."),
            InputParam.template("max_sequence_length", default=512),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="prompt_embeds",
                type_hint=torch.Tensor,
                description="Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).",
            ),
            OutputParam(
                name="prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Per-prompt boolean text mask (B, text_seq_len).",
            ),
            OutputParam(
                name="negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Per-prompt negative text features (only when guidance is enabled).",
            ),
            OutputParam(
                name="negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Per-prompt negative text mask (only when guidance is enabled).",
            ),
        ]

    def _encode_prompt(self, components, prompt, max_sequence_length, device):
        """Tokenize `prompt` into the fixed-length Krea 2 layout and tap the selected encoder hidden states.

        Mirrors `Krea2Pipeline.get_text_hidden_states`. Returns a `(hidden_states, attention_mask)` tuple of shapes
        `(batch_size, text_seq_len, num_text_layers, text_hidden_dim)` and `(batch_size, text_seq_len)` (bool).
        """
        tokenizer = components.tokenizer
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prefix_idx = _PROMPT_TEMPLATE_ENCODE_START_IDX
        text = [_PROMPT_TEMPLATE_ENCODE_PREFIX + e for e in prompt]
        text_tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length + prefix_idx - _PROMPT_TEMPLATE_ENCODE_NUM_SUFFIX_TOKENS,
            return_tensors="pt",
        ).to(device)
        suffix_tokens = tokenizer([_PROMPT_TEMPLATE_ENCODE_SUFFIX] * len(text), return_tensors="pt").to(device)

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

        # Krea 2 pads in the middle of the template (`[prefix | prompt | PAD | suffix]`), so the suffix tokens sit
        # downstream of the padding. The text features must use positions that count only real tokens (padding does
        # not consume a position) to match how the model was trained; otherwise the suffix gets a shifted mRoPE phase.
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = components.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = torch.stack([outputs.hidden_states[i] for i in KREA2_TEXT_ENCODER_SELECT_LAYERS], dim=2)

        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else list(block_state.prompt)

        block_state.prompt_embeds, block_state.prompt_embeds_mask = self._encode_prompt(
            components, prompts, block_state.max_sequence_length, device
        )

        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt
            if negative_prompt is None:
                negative_prompt = ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompts)
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = self._encode_prompt(
                components, negative_prompt, block_state.max_sequence_length, device
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2TurboTextEncoderStep(Krea2TextEncoderStep):
    """
    Text encoder step for the distilled Krea 2 turbo checkpoint that tokenizes the prompt(s) with the Krea 2 chat
    template, runs the Qwen3-VL text encoder, and stacks a fixed set of decoder-layer hidden states per token as the
    transformer's text conditioning. The distilled checkpoint runs without classifier-free guidance, so it takes no
    negative prompt and has no guider.

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. tokenizer (`AutoTokenizer`): The tokenizer paired
          with the text encoder.

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Text encoder step for the distilled Krea 2 turbo checkpoint that tokenizes the prompt(s) with the Krea 2 "
            "chat template, runs the Qwen3-VL text encoder, and stacks a fixed set of decoder-layer hidden states per "
            "token as the transformer's text conditioning. The distilled checkpoint runs without classifier-free "
            "guidance, so it takes no negative prompt and has no guider."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLModel, description="The Qwen3-VL text encoder."),
            ComponentSpec("tokenizer", AutoTokenizer, description="The tokenizer paired with the text encoder."),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam.template("max_sequence_length", default=512),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="prompt_embeds",
                type_hint=torch.Tensor,
                description="Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).",
            ),
            OutputParam(
                name="prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Per-prompt boolean text mask (B, text_seq_len).",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else list(block_state.prompt)

        block_state.prompt_embeds, block_state.prompt_embeds_mask = self._encode_prompt(
            components, prompts, block_state.max_sequence_length, device
        )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2ReferenceTextEncoderStep(ModularPipelineBlocks):
    """
    Encode prompts together with a reference image through Qwen3-VL for reference-conditioned Krea 2 generation.

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. reference_image_processor
          (`Krea2ReferenceImageProcessor`): The Qwen3-VL processor used for image-grounded prompt encoding. tokenizer
          (`AutoTokenizer`): The tokenizer paired with the text encoder. guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The negative prompt(s) for CFG.
          reference_image (`Image | list`):
              First reference image(s), or scene reference for two-reference generation.
          reference_image_2 (`Image | list`, *optional*):
              Optional second reference image(s), used as the subject reference.
          reference_image_encoder_resolution (`int`, *optional*, defaults to 768):
              Maximum reference-image side length used by the Qwen3-VL encoder. Use 0 for native resolution.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Encode prompts together with a reference image through Qwen3-VL for reference-conditioned Krea 2 generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLModel, description="The Qwen3-VL text encoder."),
            ComponentSpec(
                "reference_image_processor",
                Krea2ReferenceImageProcessor,
                default_creation_method="from_config",
                description="The Qwen3-VL processor used for image-grounded prompt encoding.",
            ),
            ComponentSpec("tokenizer", AutoTokenizer, description="The tokenizer paired with the text encoder."),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.5, "use_original_formulation": True}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam(name="negative_prompt", type_hint=str, description="The negative prompt(s) for CFG."),
            InputParam(
                name="reference_image",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                required=True,
                description="First reference image(s), or scene reference for two-reference generation.",
            ),
            InputParam(
                name="reference_image_2",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                description="Optional second reference image(s), used as the subject reference.",
            ),
            InputParam(
                name="reference_image_encoder_resolution",
                type_hint=int,
                default=768,
                description="Maximum reference-image side length used by the Qwen3-VL encoder. Use 0 for native resolution.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
        ]

    def _encode_prompt(self, components, prompts, reference_images, reference_images_2, encoder_resolution, device):
        references_by_input = []
        for name, images in (("reference_image", reference_images), ("reference_image_2", reference_images_2)):
            if images is None:
                continue
            if isinstance(images, PIL.Image.Image):
                images = [images]
            if len(images) == 1 and len(prompts) > 1:
                images = images * len(prompts)
            if len(images) != len(prompts):
                raise ValueError(
                    f"`{name}` must contain one image or one image per prompt, but got {len(images)} images for "
                    f"{len(prompts)} prompts."
                )
            references_by_input.append(images)

        processed_images = []
        for prompt_images in zip(*references_by_input):
            for image in prompt_images:
                image = image.convert("RGB")
                if encoder_resolution and max(image.size) > encoder_resolution:
                    scale = encoder_resolution / max(image.size)
                    image = image.resize(
                        (max(16, round(image.width * scale)), max(16, round(image.height * scale))),
                        PIL.Image.Resampling.LANCZOS,
                    )
                processed_images.append(image)

        image_inputs = components.reference_image_processor(images=processed_images, return_tensors="pt")
        image_token = "<|image_pad|>"
        image_token_counts = (
            image_inputs.image_grid_thw.prod(dim=1) // components.reference_image_processor.merge_size**2
        ).tolist()
        num_references = len(references_by_input)
        vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
        texts = []
        for prompt_index, prompt in enumerate(prompts):
            prompt_vision_blocks = ""
            for reference_index in range(num_references):
                count = image_token_counts[prompt_index * num_references + reference_index]
                prompt_vision_blocks += vision_block.replace(image_token, image_token * count)
            texts.append(_REFERENCE_PROMPT_TEMPLATE.format(prompt_vision_blocks, prompt))
        text_inputs = components.tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        image_token_id = components.tokenizer.convert_tokens_to_ids(image_token)
        outputs = components.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=image_inputs.pixel_values.to(device),
            image_grid_thw=image_inputs.image_grid_thw.to(device),
            mm_token_type_ids=input_ids.eq(image_token_id).long(),
            output_hidden_states=True,
        )
        hidden_states = torch.stack([outputs.hidden_states[i] for i in KREA2_TEXT_ENCODER_SELECT_LAYERS], dim=2)
        return (
            hidden_states[:, _PROMPT_TEMPLATE_ENCODE_START_IDX:],
            attention_mask[:, _PROMPT_TEMPLATE_ENCODE_START_IDX:].bool(),
        )

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else list(block_state.prompt)

        block_state.prompt_embeds, block_state.prompt_embeds_mask = self._encode_prompt(
            components,
            prompts,
            block_state.reference_image,
            block_state.reference_image_2,
            block_state.reference_image_encoder_resolution,
            device,
        )

        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        if components.requires_unconditional_embeds:
            negative_prompts = block_state.negative_prompt
            if negative_prompts is None:
                negative_prompts = ""
            if isinstance(negative_prompts, str):
                negative_prompts = [negative_prompts] * len(prompts)
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = self._encode_prompt(
                components,
                negative_prompts,
                block_state.reference_image,
                block_state.reference_image_2,
                block_state.reference_image_encoder_resolution,
                device,
            )
            prompt_length = block_state.prompt_embeds.shape[1]
            negative_length = block_state.negative_prompt_embeds.shape[1]
            if prompt_length < negative_length:
                padding = negative_length - prompt_length
                block_state.prompt_embeds = torch.nn.functional.pad(
                    block_state.prompt_embeds, (0, 0, 0, 0, 0, padding)
                )
                block_state.prompt_embeds_mask = torch.nn.functional.pad(
                    block_state.prompt_embeds_mask, (0, padding), value=False
                )
            elif negative_length < prompt_length:
                padding = prompt_length - negative_length
                block_state.negative_prompt_embeds = torch.nn.functional.pad(
                    block_state.negative_prompt_embeds, (0, 0, 0, 0, 0, padding)
                )
                block_state.negative_prompt_embeds_mask = torch.nn.functional.pad(
                    block_state.negative_prompt_embeds_mask, (0, padding), value=False
                )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2TurboReferenceTextEncoderStep(Krea2ReferenceTextEncoderStep):
    """
    Encode prompts with a reference image for reference-conditioned Krea 2 Turbo generation.

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. reference_image_processor
          (`Krea2ReferenceImageProcessor`): The Qwen3-VL processor used for image-grounded prompt encoding. tokenizer
          (`AutoTokenizer`): The tokenizer paired with the text encoder.

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          reference_image (`Image | list`):
              First reference image(s), or scene reference for two-reference generation.
          reference_image_2 (`Image | list`, *optional*):
              Optional second reference image(s), used as the subject reference.
          reference_image_encoder_resolution (`int`, *optional*, defaults to 768):
              Maximum reference-image side length used by the Qwen3-VL encoder. Use 0 for native resolution.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
    """

    @property
    def description(self) -> str:
        return "Encode prompts with a reference image for reference-conditioned Krea 2 Turbo generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLModel, description="The Qwen3-VL text encoder."),
            ComponentSpec(
                "reference_image_processor",
                Krea2ReferenceImageProcessor,
                default_creation_method="from_config",
                description="The Qwen3-VL processor used for image-grounded prompt encoding.",
            ),
            ComponentSpec("tokenizer", AutoTokenizer, description="The tokenizer paired with the text encoder."),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam(
                name="reference_image",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                required=True,
                description="First reference image(s), or scene reference for two-reference generation.",
            ),
            InputParam(
                name="reference_image_2",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                description="Optional second reference image(s), used as the subject reference.",
            ),
            InputParam(
                name="reference_image_encoder_resolution",
                type_hint=int,
                default=768,
                description="Maximum reference-image side length used by the Qwen3-VL encoder. Use 0 for native resolution.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("prompt_embeds"), OutputParam.template("prompt_embeds_mask")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        prompts = [block_state.prompt] if isinstance(block_state.prompt, str) else list(block_state.prompt)
        block_state.prompt_embeds, block_state.prompt_embeds_mask = self._encode_prompt(
            components,
            prompts,
            block_state.reference_image,
            block_state.reference_image_2,
            block_state.reference_image_encoder_resolution,
            components._execution_device,
        )
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2ProcessImagesInputStep(ModularPipelineBlocks):
    """
    Preprocess an input image for Krea 2 image-to-image generation.

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.

      Outputs:
          processed_image (`Tensor`):
              The preprocessed input image.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Preprocess an input image for Krea 2 image-to-image generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            )
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image", required=True),
            InputParam.template("height"),
            InputParam.template("width"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="processed_image", type_hint=torch.Tensor, description="The preprocessed input image.")
        ]

    @staticmethod
    def check_inputs(height, width, multiple):
        if height is not None and height % multiple != 0:
            raise ValueError(f"`height` must be divisible by {multiple}, but is {height}")
        if width is not None and width % multiple != 0:
            raise ValueError(f"`width` must be divisible by {multiple}, but is {width}")

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.height, block_state.width, components.image_processor.config.vae_scale_factor)
        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        block_state.processed_image = components.image_processor.preprocess(
            image=block_state.image, height=height, width=width
        )
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2InpaintProcessImagesInputStep(ModularPipelineBlocks):
    """
    Preprocess an input image and mask together for Krea 2 inpainting.

      Components:
          image_mask_processor (`InpaintProcessor`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          mask_image (`Image`):
              Mask image for inpainting.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.

      Outputs:
          processed_image (`Tensor`):
              The preprocessed input image.
          processed_mask_image (`Tensor`):
              The preprocessed inpainting mask.
          mask_overlay_kwargs (`dict`):
              Arguments used to overlay a cropped inpainting result on the original image.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Preprocess an input image and mask together for Krea 2 inpainting."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            )
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image", required=True),
            InputParam.template("mask_image", required=True),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("padding_mask_crop"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="processed_image", type_hint=torch.Tensor, description="The preprocessed input image."),
            OutputParam(
                name="processed_mask_image", type_hint=torch.Tensor, description="The preprocessed inpainting mask."
            ),
            OutputParam(
                name="mask_overlay_kwargs",
                type_hint=dict,
                description="Arguments used to overlay a cropped inpainting result on the original image.",
            ),
        ]

    @staticmethod
    def check_inputs(height, width, multiple):
        if height is not None and height % multiple != 0:
            raise ValueError(f"`height` must be divisible by {multiple}, but is {height}")
        if width is not None and width % multiple != 0:
            raise ValueError(f"`width` must be divisible by {multiple}, but is {width}")

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(
            block_state.height, block_state.width, components.image_mask_processor.config.vae_scale_factor
        )
        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        block_state.processed_image, block_state.processed_mask_image, block_state.mask_overlay_kwargs = (
            components.image_mask_processor.preprocess(
                image=block_state.image,
                mask=block_state.mask_image,
                height=height,
                width=width,
                padding_mask_crop=block_state.padding_mask_crop,
            )
        )
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2VaeEncoderStep(ModularPipelineBlocks):
    """
    Encode a preprocessed image into normalized Krea 2 image latents.

      Components:
          vae (`AutoencoderKLQwenImage`)

      Inputs:
          processed_image (`Tensor`):
              The preprocessed image.

      Outputs:
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Encode a preprocessed image into normalized Krea 2 image latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLQwenImage)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="processed_image", required=True, type_hint=torch.Tensor, description="The preprocessed image."
            )
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("image_latents")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        image = block_state.processed_image
        if image.ndim == 4:
            image = image.unsqueeze(2)
        elif image.ndim != 5:
            raise ValueError(f"`processed_image` must have 4 or 5 dimensions, but got {image.ndim}")

        image = image.to(device=components._execution_device, dtype=components.vae.dtype)
        image_latents = components.vae.encode(image).latent_dist.mode()

        latents_mean = torch.tensor(components.vae.config.latents_mean).view(1, components.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std).view(1, components.vae.config.z_dim, 1, 1, 1)
        latents_mean = latents_mean.to(image_latents.device, image_latents.dtype)
        latents_std = latents_std.to(image_latents.device, image_latents.dtype)
        block_state.image_latents = (image_latents - latents_mean) / latents_std

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2ReferenceProcessImagesInputStep(ModularPipelineBlocks):
    """
    Preprocess a reference image at the target output resolution for VAE encoding.

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          reference_image (`Image | list`):
              First reference image(s), or scene reference for two-reference generation.
          reference_image_2 (`Image | list`, *optional*):
              Optional second reference image(s), used as the subject reference.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.

      Outputs:
          processed_reference_images (`list`):
              Reference images resized and normalized for VAE encoding in conditioning order.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Preprocess a reference image at the target output resolution for VAE encoding."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            )
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="reference_image",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                required=True,
                description="First reference image(s), or scene reference for two-reference generation.",
            ),
            InputParam(
                name="reference_image_2",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                description="Optional second reference image(s), used as the subject reference.",
            ),
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="processed_reference_images",
                type_hint=list[torch.Tensor],
                description="Reference images resized and normalized for VAE encoding in conditioning order.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        multiple = components.image_processor.config.vae_scale_factor
        if block_state.height % multiple != 0 or block_state.width % multiple != 0:
            raise ValueError(f"`height` and `width` must be divisible by {multiple} for reference conditioning.")
        reference_images = [block_state.reference_image]
        if block_state.reference_image_2 is not None:
            reference_images.append(block_state.reference_image_2)
        block_state.processed_reference_images = [
            components.image_processor.preprocess(image=image, height=block_state.height, width=block_state.width)
            for image in reference_images
        ]
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2ReferenceVaeEncoderStep(ModularPipelineBlocks):
    """
    Encode a preprocessed reference image into normalized Krea 2 latents.

      Components:
          vae (`AutoencoderKLQwenImage`)

      Inputs:
          processed_reference_images (`list`):
              The preprocessed reference images in conditioning order.

      Outputs:
          reference_image_latents (`list`):
              Normalized latent representations of the reference images in conditioning order.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Encode a preprocessed reference image into normalized Krea 2 latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLQwenImage)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="processed_reference_images",
                type_hint=list[torch.Tensor],
                required=True,
                description="The preprocessed reference images in conditioning order.",
            )
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="reference_image_latents",
                type_hint=list[torch.Tensor],
                description="Normalized latent representations of the reference images in conditioning order.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        latents_mean = torch.tensor(components.vae.config.latents_mean).view(1, components.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std).view(1, components.vae.config.z_dim, 1, 1, 1)
        block_state.reference_image_latents = []
        for processed_reference_image in block_state.processed_reference_images:
            reference_image = processed_reference_image.unsqueeze(2).to(
                device=components._execution_device, dtype=components.vae.dtype
            )
            reference_image_latents = components.vae.encode(reference_image).latent_dist.mode()
            reference_image_latents = (
                reference_image_latents - latents_mean.to(reference_image_latents)
            ) / latents_std.to(reference_image_latents)
            block_state.reference_image_latents.append(reference_image_latents)
        self.set_block_state(state, block_state)
        return components, state
