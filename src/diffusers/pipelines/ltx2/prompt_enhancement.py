# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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

from typing import Any

import PIL.Image
import torch

from ...image_processor import PipelineImageInput
from .utils import GEMMA3_PROMPT_ENHANCEMENT_CONFIG, GEMMA4_PROMPT_ENHANCEMENT_CONFIG


# Matches ltx-pipelines `generate_enhanced_prompt` / `clean_response`.
_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")
_ENHANCE_IMAGE_LONG_SIDE = 896


def clean_response(text: str) -> str:
    """Clean curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Left-pad sequence length to a multiple of `alignment` for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len
    if padding_length <= 0:
        return model_inputs

    def _left_pad(tensor: torch.Tensor, value: int | float) -> torch.Tensor:
        pad = torch.full((1, padding_length), value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([pad, tensor], dim=1)

    model_inputs["input_ids"] = _left_pad(model_inputs.input_ids, pad_token_id)
    model_inputs["attention_mask"] = _left_pad(model_inputs.attention_mask, 0)
    if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
        model_inputs["token_type_ids"] = _left_pad(model_inputs["token_type_ids"], 0)
    return model_inputs


def _prepare_enhance_image(image: PipelineImageInput, long_side: int = _ENHANCE_IMAGE_LONG_SIDE) -> PIL.Image.Image:
    """Resize a reference image so its long side is `long_side`, matching ltx-pipelines enhance prep."""
    if isinstance(image, PIL.Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise ValueError(
            f"Image-conditioned prompt enhancement requires a `PIL.Image.Image`, got {type(image)}. "
            "Convert the reference frame to PIL before enabling enhancement."
        )
    width, height = pil_image.size
    scale = long_side / float(max(width, height))
    target_width = int(width * scale)
    target_height = int(height * scale)
    if (target_width, target_height) != (width, height):
        pil_image = pil_image.resize((target_width, target_height), resample=PIL.Image.Resampling.BICUBIC)
    return pil_image


class LTX2PromptEnhancementMixin:
    """Shared Gemma prompt-enhancement helpers for LTX-2 pipelines."""

    @torch.no_grad()
    def enhance_prompt(
        self,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int | None = None,
        seed: int = 10,
        generator: torch.Generator | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        image: PipelineImageInput | None = None,
    ):
        """
        Enhances the supplied `prompt` by generating a new prompt using the prompt enhancer (a Gemma
        conditional-generation model) from it and a system prompt. When `image` is supplied, the enhancer is also
        conditioned on that reference frame (I2V / keyframe-style enhancement). Uses the dedicated `prompt_enhancer`
        component if one is configured (e.g. LTX-2.5, whose text encoder isn't trained for enhancement), otherwise
        falls back to the main `text_encoder` (LTX-2.0/2.3, which double as their own enhancer).

        Message templates, decoding kwargs, response cleaning, and image long-side prep match `ltx-core` /
        `ltx-pipelines` (`enhance_t2v` / `enhance_i2v` / `generate_enhanced_prompt`).
        """
        device = device or self._execution_device
        using_dedicated_enhancer = getattr(self, "prompt_enhancer", None) is not None
        enhancer = self.prompt_enhancer if using_dedicated_enhancer else self.text_encoder
        config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG if using_dedicated_enhancer else GEMMA3_PROMPT_ENHANCEMENT_CONFIG

        generation_kwargs = (
            dict(generation_kwargs) if generation_kwargs is not None else dict(config.generation_kwargs)
        )
        if max_new_tokens is None:
            max_new_tokens = config.max_new_tokens

        # Templates match ltx-core `LTXGemmaTextEncoder.enhance_t2v` / `enhance_i2v` for both Gemma 3 and 4.
        if image is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"user prompt: {prompt}"},
            ]
            enhance_image = None
        else:
            enhance_image = _prepare_enhance_image(image)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                    ],
                },
            ]

        template = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.processor(text=template, images=enhance_image, return_tensors="pt").to(device)
        pad_token_id = (
            self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        )
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)
        enhancer.to(device)

        # `transformers.GenerationMixin.generate` does not support using a `torch.Generator` to control randomness,
        # so manually apply a seed for reproducible generation.
        if generator is not None:
            seed = generator.initial_seed() if not isinstance(generator, list) else generator[0].initial_seed()
        torch.manual_seed(seed)
        generated_sequences = enhancer.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )  # tensor of shape [batch_size, seq_len]

        generated_ids = [seq[len(model_inputs.input_ids[i]) :] for i, seq in enumerate(generated_sequences)]
        enhanced_prompt = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [clean_response(text) for text in enhanced_prompt]
