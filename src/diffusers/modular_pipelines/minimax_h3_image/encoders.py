# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

import numpy as np
import PIL.Image
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...models import AutoencoderKLMiniMaxH3Image
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.modular_pipelines.minimax_h3.encoders.get_qwen3vl_prompt_embeds
def get_qwen3vl_prompt_embeds(
    text_encoder,
    processor,
    token_ids: list[int],
    vision_inputs: dict | None = None,
    text_encoder_layer: int = 50,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""
    Encode one tokenized MiniMax-H3 presentation with the Qwen3-VL conditioner.

    Unlike its `get_<encoder>_prompt_embeds` siblings this takes the *tokenized* presentation rather than a prompt
    string: MiniMax-H3's tokenization is the presentation-building phase that precedes it, where every reference or
    keyframe prepends a label and a vision block. The `mm_token_type_ids` built here are Qwen-internal (`0` text, `1`
    image, `2` video — they drive Qwen3-VL's per-modality-run rotary layout and never leave this function) and are not
    MiniMax-H3's own per-row modality tags, which the presentation phase produces alongside `token_ids`.

    Args:
        text_encoder (`Qwen3VLForConditionalGeneration`): The conditioner.
        processor (`Qwen3VLProcessor`): Its processor, which derives the token type ids from the vision pad ids.
        token_ids (`list[int]`): The tokenized presentation.
        vision_inputs (`dict`, *optional*):
            The vision tensors of the presentation's blocks, by the conditioner's own parameter names — `pixel_values`
            / `image_grid_thw` for images, `pixel_values_videos` / `video_grid_thw` for videos.
        text_encoder_layer (`int`, *optional*, defaults to 50):
            Which hidden state conditions the transformer, i.e. `components.text_encoder_layer`.
        device (`torch.device`, *optional*): The device to run the conditioner on.
        dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

    Returns:
        `torch.Tensor`: the `(1, num_text_tokens, 5120)` hidden state after decoder layer `text_encoder_layer`.
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers <= text_encoder_layer:
        raise ValueError(
            f"MiniMax-H3 conditions on `hidden_states[{text_encoder_layer}]` of its Qwen3-VL "
            f"conditioner, which needs more than {text_encoder_layer} decoder layers, but "
            f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
            f"{text_encoder_layer} layers is post-norm and is not the conditioning MiniMax-H3 expects."
        )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)

    vision_kwargs = {}
    for name, value in (vision_inputs or {}).items():
        vision_kwargs[name] = value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)

    # `text_encoder.model` is a submodule, and a CPU-offload hook — accelerate's or the one the
    # `ComponentsManager` attaches — wraps the *top-level* module's `forward` alone, so calling the submodule
    # directly would leave the conditioner on the CPU. Fire the hook by hand instead of routing through
    # `text_encoder(...)`: MiniMax-H3 reads `hidden_states[50]` and never uses the language-model head, whose
    # vocabulary-wide projection over every token is all the top-level forward would add.
    # TODO: firing another module's offload hook by hand is not something a block should do. It is here
    # because MiniMax-H3 reads `hidden_states[50]` off `text_encoder.model` while the hook wraps only the
    # top-level `forward`. Needs a real answer — an opt-in on the conditioner, or a hook that follows
    # submodule calls.
    hook = getattr(text_encoder, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(text_encoder)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        **vision_kwargs,
    )
    return outputs.hidden_states[text_encoder_layer].to(device=device, dtype=dtype)


class MiniMaxH3ImageTextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Encodes the prompt of a `text2image` request: verbatim, with no chat template and no special tokens, "
            "truncated to `max_sequence_length` tokens. The checkpoint is guidance-distilled, so there is no negative "
            "prompt and no unconditional branch."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("processor", Qwen3VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template(
                "prompt_embeds",
                description=(
                    "The hidden state the transformer conditions on, of shape `(1, num_text_tokens, 5120)`, read "
                    "after the 50th decoder layer of the Qwen3-VL conditioner."
                ),
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description="The modality tag of every row of `prompt_embeds`, all text for this presentation.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str) or not block_state.prompt:
            raise ValueError(
                "MiniMax-H3-Image packs one request into one sequence, so `prompt` must be a single non-empty string, "
                f"got {block_state.prompt!r}."
            )

        token_ids = components.tokenizer(block_state.prompt, add_special_tokens=False)["input_ids"]
        token_ids = token_ids[: components.max_sequence_length]

        block_state.prompt_embeds = get_qwen3vl_prompt_embeds(
            components.text_encoder,
            components.processor,
            token_ids,
            {},
            text_encoder_layer=components.text_encoder_layer,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )
        block_state.text_token_tags = torch.full((len(token_ids),), components.text_tag, dtype=torch.long)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageEditTextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Encodes the presentation of an edit: a `Picture i: ` label and a vision block per reference image, then "
            "the prompt verbatim, with no chat template and no special tokens. The prompt and every label are "
            "truncated to `max_sequence_length` tokens. The checkpoint is guidance-distilled, so there is no negative "
            "prompt and no unconditional branch."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("processor", Qwen3VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
            InputParam(
                name="vision_images",
                type_hint=list[PIL.Image.Image],
                required=True,
                description="The reference images resized for the text encoder.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template(
                "prompt_embeds",
                description=(
                    "The hidden state the transformer conditions on, of shape `(1, num_text_tokens, 5120)`, read "
                    "after the 50th decoder layer of the Qwen3-VL conditioner."
                ),
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description="The modality tag of every row of `prompt_embeds`; a vision block is tagged as image.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str) or not block_state.prompt:
            raise ValueError(
                "MiniMax-H3-Image packs one request into one sequence, so `prompt` must be a single non-empty string, "
                f"got {block_state.prompt!r}."
            )
        tokenizer, processor = components.tokenizer, components.processor
        text_tag, image_tag = components.text_tag, components.image_tag
        max_sequence_length = components.max_sequence_length

        vision = processor.image_processor(images=block_state.vision_images, return_tensors="pt")
        merge_size = processor.image_processor.merge_size**2

        # The label of a reference carries no angle brackets here, unlike the `<Picture i>` the prompt refers to it by.
        token_ids, token_tags = [], []
        for index, grid in enumerate(vision["image_grid_thw"]):
            label_ids = tokenizer(f"Picture {index + 1}: ", add_special_tokens=False)["input_ids"]
            label_ids = label_ids[:max_sequence_length]
            vision_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge_size)
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label_ids + vision_ids
            token_tags += [text_tag] * len(label_ids) + [image_tag] * len(vision_ids)
        prompt_ids = tokenizer(block_state.prompt, add_special_tokens=False)["input_ids"][:max_sequence_length]
        token_ids += prompt_ids
        token_tags += [text_tag] * len(prompt_ids)

        block_state.prompt_embeds = get_qwen3vl_prompt_embeds(
            components.text_encoder,
            processor,
            token_ids,
            {"pixel_values": vision["pixel_values"], "image_grid_thw": vision["image_grid_thw"]},
            text_encoder_layer=components.text_encoder_layer,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )
        block_state.text_token_tags = torch.tensor(token_tags, dtype=torch.long)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageEditVaeEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Encodes the reference images of an edit into conditioning latents. Each posterior is sampled off the "
            "request's generator, rounded to float16, cropped to whole patches and normalized."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLMiniMaxH3Image)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="reference_images",
                type_hint=list[PIL.Image.Image],
                required=True,
                description="The reference images resized for the VAE.",
            ),
            InputParam.template(
                "generator",
                description="The generator of the request. One posterior sample per reference is drawn from it.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The conditioning latents of the reference images, one `(1, latent_channels, latent_height, "
                    "latent_width)` tensor per reference, each at the reference's own size."
                ),
            )
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device
        ratio = components.vae_spatial_compression_ratio
        _, patch_h, patch_w = components.patch_size
        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1)
        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1)

        block_state.reference_latents = []
        for image in block_state.reference_images:
            pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None]
            # A reference keeps its own size, so it is cropped at the top left to whole VAE cells.
            pixels = pixels[..., : pixels.shape[-2] // ratio * ratio, : pixels.shape[-1] // ratio * ratio]
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            posterior = components.vae.encode(pixels, return_dict=False)[0]
            # The released recipe: a posterior sample rounded to float16, then cropped to whole patches.
            latents = posterior.sample(generator=block_state.generator).to(torch.float16).float()
            latents = latents[..., : latents.shape[-2] // patch_h * patch_h, : latents.shape[-1] // patch_w * patch_w]
            block_state.reference_latents.append((latents - latents_mean) / latents_std)

        self.set_block_state(state, block_state)
        return components, state
