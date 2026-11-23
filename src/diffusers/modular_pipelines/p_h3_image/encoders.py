# Copyright 2026 Pruna AI, The MiniMax and HuggingFace Teams. All rights reserved.
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

from ...models import AutoencoderKLPH3Image, PH3ImageTextConnector
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import PH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.modular_pipelines.minimax_h3.encoders.encode_vae_condition with AutoencoderKLMiniMaxH3->AutoencoderKLPH3Image
def encode_vae_condition(
    vae, pixels: torch.Tensor, pixel_mean: tuple, pixel_std: tuple, encode_seed: int = 42
) -> torch.Tensor:
    r"""
    Encode one visual condition — a keyframe or a `ref2va` reference — into normalized conditioning latents.

    The recipe is the released model's, and every part of it is needed to reproduce its conditioning: the pixels are
    ImageNet-normalized, the posterior is *sampled* under a fresh generator seeded independently of the request, and
    the sampled latent is rounded to float16 (~11 bits of every conditioning latent) before being normalized. A single
    frame is encoded by the (tiled) spatial encoder alone; a frame stack goes through the temporal chunking, which is
    what turns `17 * n + 5` frames into `5 * n + 2` latent frames.

    Args:
        vae (`AutoencoderKLPH3Image`): The video VAE.
        pixels (`torch.Tensor` of shape `(1, 3, num_frames, height, width)`):
            The condition's `uint8` pixels, on the VAE's device. `num_frames` is 1 for a keyframe or an image
            reference.
        pixel_mean (`tuple[float, float, float]`), pixel_std (`tuple[float, float, float]`):
            The video VAE's pixel convention, i.e. `components.pixel_mean` / `components.pixel_std`.
        encode_seed (`int`, *optional*, defaults to 42):
            Seed the posterior is sampled under, i.e. `components.keyframe_encode_seed`.

    Returns:
        `torch.Tensor`: one `(1, latent_channels, num_latent_frames, latent_height, latent_width)` float32 CPU tensor.
    """
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)

    pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
    posterior = vae.encode(pixels, return_dict=False)[0]
    latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
    latents = latents.to(torch.float16).float().cpu()
    return (latents - latents_mean) / latents_std


def get_p_h3_image_prompt_embeds(
    text_encoder,
    text_connector,
    processor,
    token_ids: list[int],
    vision_inputs: dict | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""
    Encode one tokenized P-H3-Image presentation: the hidden states of the text encoder, mapped to the transformer's
    text conditioning by the text connector.

    Args:
        text_encoder (`Qwen3VLForConditionalGeneration`): The text encoder.
        text_connector (`PH3ImageTextConnector`): Maps the text encoder's hidden states to the conditioning.
        processor (`Qwen3VLProcessor`): The text encoder's processor, used for the token type ids.
        token_ids (`list[int]`): The tokenized presentation.
        vision_inputs (`dict`, *optional*):
            The `pixel_values` and `image_grid_thw` of the presentation's image.
        device (`torch.device`, *optional*): The device to run the text encoder on.
        dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

    Returns:
        `torch.Tensor`: the `(1, num_text_tokens, 5120)` conditioning.
    """
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)

    vision_kwargs = {}
    for name, value in (vision_inputs or {}).items():
        vision_kwargs[name] = value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)

    # Only the hidden states are used; `logits_to_keep=1` keeps the language-model head to a single token.
    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        logits_to_keep=1,
        **vision_kwargs,
    )
    return text_connector(outputs.hidden_states).to(device=device, dtype=dtype)


class PH3ImageTextEncoderStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return "Encodes the prompt of a `text2image` request, verbatim, with no chat template and no special tokens."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("text_connector", PH3ImageTextConnector),
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
                description="The conditioning MiniMax-H3's transformer reads, of shape `(1, num_text_tokens, 5120)`.",
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description=("The modality tag of every row of `prompt_embeds`, all text."),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                "P-H3-Image packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(block_state.prompt)}."
            )

        token_ids = components.tokenizer(block_state.prompt, add_special_tokens=False)["input_ids"]

        block_state.prompt_embeds = get_p_h3_image_prompt_embeds(
            components.text_encoder,
            components.text_connector,
            components.processor,
            token_ids,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )
        block_state.text_token_tags = torch.full((len(token_ids),), components.text_tag, dtype=torch.long)

        self.set_block_state(state, block_state)
        return components, state


class PH3ImageEditTextEncoderStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            'Encodes an edit as `"<Picture i>: "` and the vision tokens of every image, then the instruction after '
            '`"Edit <Picture 1>. "`, with no chat template and no special tokens. An edit brief from the prompt '
            "upsampler, or a prompt that already starts with that prefix, is encoded as written."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("text_connector", PH3ImageTextConnector),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("processor", Qwen3VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt", description="The prompt, or the edit instruction when `image` is passed. A single string."
            ),
            InputParam(
                name="vision_images",
                type_hint=list[PIL.Image.Image],
                required=True,
                description="The images of the edit resized for the text encoder, the image to edit first.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template(
                "prompt_embeds",
                description="The conditioning MiniMax-H3's transformer reads, of shape `(1, num_text_tokens, 5120)`.",
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description=("The modality tag of every row of `prompt_embeds`; vision tokens are tagged as video."),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                "P-H3-Image packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(block_state.prompt)}."
            )
        tokenizer, processor = components.tokenizer, components.processor
        text_tag, video_tag = components.text_tag, components.video_tag

        vision = processor.image_processor(images=block_state.vision_images, return_tensors="pt")
        merge_size = processor.image_processor.merge_size**2
        token_ids, token_tags = [], []
        for index, grid in enumerate(vision["image_grid_thw"]):
            label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge_size)
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label_ids + vision_ids
            token_tags += [text_tag] * len(label_ids) + [video_tag] * len(vision_ids)
        prompt = block_state.prompt
        if not prompt.lstrip().startswith(("Edit <Picture 1>.", "subject_definitions:")):
            prompt = f"Edit <Picture 1>. {prompt}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [text_tag] * len(prompt_ids)

        block_state.prompt_embeds = get_p_h3_image_prompt_embeds(
            components.text_encoder,
            components.text_connector,
            processor,
            token_ids,
            {"pixel_values": vision["pixel_values"], "image_grid_thw": vision["image_grid_thw"]},
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )
        block_state.text_token_tags = torch.tensor(token_tags, dtype=torch.long)

        self.set_block_state(state, block_state)
        return components, state


class PH3ImageEditVaeEncoderStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            "Encodes the images of an edit into conditioning latents with MiniMax-H3's conditioning recipe: a seeded "
            "posterior sample, rounded to float16 and normalized."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLPH3Image)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="reference_images",
                type_hint=list[PIL.Image.Image],
                required=True,
                description="The images of the edit resized for the VAE, the image to edit first.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The conditioning latents of the images of the edit, one `(1, latent_channels, 1, latent_height, "
                    "latent_width)` tensor per image."
                ),
            )
        ]

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.condition_latents = [
            encode_vae_condition(
                components.vae,
                torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None],
                components.pixel_mean,
                components.pixel_std,
                components.keyframe_encode_seed,
            )
            for image in block_state.reference_images
        ]

        self.set_block_state(state, block_state)
        return components, state
