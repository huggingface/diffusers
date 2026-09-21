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

import PIL.Image
import torch

from ...image_processor import PipelineImageInput


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
