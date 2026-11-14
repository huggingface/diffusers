# Copyright 2026 Pruna AI and The HuggingFace Team. All rights reserved.
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
import re

import PIL.Image
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from ..modular_pipeline import AutoPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .before_encoder import MAX_EDIT_IMAGES, _size_at_area
from .modular_pipeline import PH3ImageModularPipeline


ASPECT_RATIOS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}
UPSAMPLER_IMAGE_PIXELS = 1_000_000
T2I_MAX_NEW_TOKENS = 1024
EDIT_MAX_NEW_TOKENS = 1536

EDIT_SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_SECTION_RE = re.compile(r"(?m)^(" + "|".join(EDIT_SECTIONS) + r"):[ \t]*$")
_RETENTION_RE = re.compile(
    r"^<(Picture|Subject) (\d+)>(?: \(([^)]*)\))?:\s*(fully_preserved|partially_preserved|attribute_transfer|"
    r"weak_reference)\s*[-–—]\s*(.*)$"
)


def aspect_ratio_label(width: int, height: int) -> str:
    """The aspect ratio the prompt upsampler knows that is closest to `width / height`."""
    return min(ASPECT_RATIOS, key=lambda label: abs(math.log(width / height / ASPECT_RATIOS[label])))


def is_edit_prompt(prompt: str) -> bool:
    r"""
    Whether an edit prompt has one of the two forms P-H3-Image edits from.

    A brief is the six sections of `EDIT_SECTIONS` in order, with `[keyframe completion]` once and every line of
    `retention_analysis` naming a `<Picture i>` or `<Subject i>` and how it is kept. An instruct prompt is one
    paragraph that starts with `"Edit <Picture 1>."` and says what it keeps.
    """
    text = prompt.strip()
    names = _SECTION_RE.findall(text)
    if names:
        parts = _SECTION_RE.split(text)
        body = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        retention = [line.strip() for line in body.get("retention_analysis", "").splitlines() if line.strip()]
        return (
            tuple(names) == EDIT_SECTIONS
            and text.count("[keyframe completion]") == 1
            and bool(retention)
            and all(_RETENTION_RE.match(line) for line in retention)
        )
    return text.startswith("Edit <Picture 1>.") and "Keep" in text and "\n" not in text


def prepare_upsampler_image(image: PIL.Image.Image) -> PIL.Image.Image:
    """An image as the prompt upsampler sees it: RGB at about one megapixel, sides multiples of 32 (Lanczos)."""
    image = image.convert("RGB")
    size = _size_at_area(*image.size, UPSAMPLER_IMAGE_PIXELS, 32)
    return image if image.size == size else image.resize(size, PIL.Image.Resampling.LANCZOS)


def upsample_prompt(
    prompt_upsampler: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    request: str,
    images: list[PIL.Image.Image] | None = None,
    aspect_ratio: str = "1:1",
    max_new_tokens: int = T2I_MAX_NEW_TOKENS,
) -> str:
    """
    The prompt P-H3-Image renders for `request`: a text-to-image prompt for `aspect_ratio`, or an edit prompt written
    from the prepared `images` (the image to edit first). One greedy generation of at most `max_new_tokens`; the
    upsampler reasons first and the prompt is what follows `</think>`.
    """
    if not isinstance(request, str):
        raise ValueError(
            f"The prompt upsampler takes one request at a time, so `prompt` must be a single string, got "
            f"{type(request).__name__}."
        )
    tokenizer = processor.tokenizer
    if images:
        content = []
        for index in range(len(images)):
            content += [{"type": "text", "text": f"<Picture {index + 1}>: "}, {"type": "image"}]
        content.append({"type": "text", "text": f"Edit <Picture 1>. {request.strip()}"})
    else:
        content = f"Aspect ratio: {aspect_ratio}\nRequest: {request.strip()}"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(prompt_upsampler.device)

    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    generated = prompt_upsampler.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=end_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
    ended = end_id in new_ids
    answer = tokenizer.decode(new_ids[: new_ids.index(end_id)] if ended else new_ids, skip_special_tokens=False)
    prompt = answer.partition("</think>")[2].strip()
    valid = ended and answer.count("</think>") == 1 and len(prompt.split()) >= 5
    if images:
        valid = valid and is_edit_prompt(prompt)
    if not valid:
        raise RuntimeError(
            "The prompt upsampler did not return a usable prompt "
            f"({'it was cut off at max_new_tokens' if not ended else 'it does not have the expected form'}):\n"
            f"{answer}\nRaise `max_new_tokens`, or remove the `prompt_upsampler` step to render the prompt as given."
        )
    return prompt


class PH3ImageText2ImagePromptUpsampleStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            "Expands a text-to-image request into the full prompt P-H3-Image is meant to render, written by the "
            "prompt upsampler for the aspect ratio of `height` and `width`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("prompt_upsampler", Qwen3_5ForConditionalGeneration, description="The prompt upsampler."),
            ComponentSpec("prompt_upsampler_processor", AutoProcessor, description="Its processor."),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt", description="The prompt, or the edit instruction when `image` is passed. A single string."
            ),
            InputParam.template("height", default=2912),
            InputParam.template("width", default=2912),
            InputParam(
                "max_new_tokens",
                type_hint=int,
                default=T2I_MAX_NEW_TOKENS,
                description="The most tokens the upsampler may generate, its reasoning included.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("prompt", type_hint=str, description="The full prompt, replacing the request.")]

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.prompt = upsample_prompt(
            components.prompt_upsampler,
            components.prompt_upsampler_processor,
            request=block_state.prompt,
            aspect_ratio=aspect_ratio_label(block_state.width, block_state.height),
            max_new_tokens=block_state.max_new_tokens,
        )
        self.set_block_state(state, block_state)
        return components, state


class PH3ImageEditPromptUpsampleStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            "Expands an edit instruction into the edit prompt P-H3-Image is meant to render, written by the prompt "
            "upsampler from the images of the edit (`image`): a brief of what to keep and what to change."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("prompt_upsampler", Qwen3_5ForConditionalGeneration, description="The prompt upsampler."),
            ComponentSpec("prompt_upsampler_processor", AutoProcessor, description="Its processor."),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt", description="The prompt, or the edit instruction when `image` is passed. A single string."
            ),
            InputParam.template(
                "image",
                description=f"The images of the edit, at most {MAX_EDIT_IMAGES}, the image to edit first; the same ones "
                "the pipeline receives.",
            ),
            InputParam(
                "max_new_tokens",
                type_hint=int,
                default=EDIT_MAX_NEW_TOKENS,
                description="The most tokens the upsampler may generate, its reasoning included.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("prompt", type_hint=str, description="The edit prompt, replacing the instruction.")]

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        images = block_state.image if isinstance(block_state.image, list) else [block_state.image]
        if len(images) > MAX_EDIT_IMAGES:
            raise ValueError(f"The prompt upsampler takes at most {MAX_EDIT_IMAGES} images, got {len(images)}.")
        block_state.prompt = upsample_prompt(
            components.prompt_upsampler,
            components.prompt_upsampler_processor,
            request=block_state.prompt,
            images=[prepare_upsampler_image(image) for image in images],
            max_new_tokens=block_state.max_new_tokens,
        )
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class PH3ImageAutoPromptUpsampleStep(AutoPipelineBlocks):
    """
    Prompt upsampling step:
       expands `prompt` into the full prompt P-H3-Image is meant to render, with the prompt upsampler (based on
       Qwen3.5-9B). The task follows the inputs, like the rest of the pipeline:
       - `PH3ImageEditPromptUpsampleStep` is used when `image` is provided (an edit, written from the images).
       - `PH3ImageText2ImagePromptUpsampleStep` otherwise (text-to-image).
      To render `prompt` as given, remove the step from the blocks: `blocks.sub_blocks.pop("prompt_upsampler")`.

      Components:
          prompt_upsampler (`Qwen3_5ForConditionalGeneration`): The prompt upsampler. prompt_upsampler_processor
          (`AutoProcessor`): Its processor.

      Inputs:
          prompt (`str`):
              The prompt, or the edit instruction when `image` is passed. A single string.
          image (`Image | list`, *optional*, depends on the workflow):
              The images of the edit, at most 8, the image to edit first; the same ones the pipeline receives.
          max_new_tokens (`int`, *optional*, depends on the workflow):
              The most tokens the upsampler may generate, its reasoning included.
          height (`int`, *optional*, defaults to 2912):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 2912):
              The width in pixels of the generated image.

      Outputs:
          prompt (`str`):
              The edit prompt, replacing the instruction.
    """

    model_name = "p-h3-image"
    block_classes = [PH3ImageEditPromptUpsampleStep, PH3ImageText2ImagePromptUpsampleStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image", None]

    @property
    def description(self) -> str:
        return (
            "Prompt upsampling step: expands `prompt` into the full prompt P-H3-Image is meant to render, with the "
            "prompt upsampler (based on Qwen3.5-9B). The task follows the inputs, like the rest of the pipeline:\n"
            " - `PH3ImageEditPromptUpsampleStep` is used when `image` is provided (an edit, written from the "
            "images).\n"
            " - `PH3ImageText2ImagePromptUpsampleStep` otherwise (text-to-image).\n"
            'To render `prompt` as given, remove the step from the blocks: `blocks.sub_blocks.pop("prompt_upsampler")`.'
        )
