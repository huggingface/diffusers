# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
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
"""Shared test fixtures for the DreamLite pipelines.

``DreamLitePipeline`` and its distilled sibling ``DreamLiteMobilePipeline`` take the exact same components, so the
tiny Qwen3-VL text encoder and the rest of the dummy component set live here and are shared by both test files.
"""

import torch
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderTiny,
    DreamLiteUNetModel,
    FlowMatchEulerDiscreteScheduler,
)

from ..testing_utils import BasePipelineTesterConfig


# Match the tiny text encoder hidden size below; the UNet's cross-attention
# dimension must match what ``encode_prompt`` returns.
CROSS_ATTN_DIM = 16


def build_tiny_text_encoder() -> Qwen3VLForConditionalGeneration:
    """Build a tiny but functional Qwen3-VL model for the fast test fixture.

    Mirrors the recipe used by ``tests/pipelines/nucleusmoe_image``: small text
    + vision configs that still go through the real Qwen3-VL forward path, so
    DreamLite's ``encode_prompt`` (chat template + tokenizer + multimodal
    processor) is exercised for real.
    """
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": CROSS_ATTN_DIM,
            "intermediate_size": CROSS_ATTN_DIM,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_scaling": {
                "mrope_section": [1, 1, 2],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000.0,
            "vocab_size": 151936,
            "head_dim": 8,
        },
        vision_config={
            "depth": 2,
            "hidden_size": CROSS_ATTN_DIM,
            "intermediate_size": CROSS_ATTN_DIM,
            "num_heads": 2,
            "out_channels": CROSS_ATTN_DIM,
            # ``out_hidden_size`` is the dim that vision tokens are projected to before
            # being merged into the text stream; it must match ``text_config.hidden_size``.
            "out_hidden_size": CROSS_ATTN_DIM,
            # Match the cached ``hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration``
            # image processor (``patch_size=14``); otherwise the pixel_values
            # produced by the processor cannot be reshaped to the model's
            # vision patch embed.
            "patch_size": 14,
        },
    )
    return Qwen3VLForConditionalGeneration(config).eval()


class DreamLiteBaseTesterConfig(BasePipelineTesterConfig):
    """Component set shared by ``DreamLitePipeline`` and ``DreamLiteMobilePipeline``."""

    # DreamLite samples its own noise; `latents` cannot be supplied by the caller.
    optional_input_params = BasePipelineTesterConfig.optional_input_params - {"latents"}
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = DreamLiteUNetModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=(
                "DreamLiteCrossAttnNoSelfAttnDownBlock2D",
                "DreamLiteCrossAttnDownBlock2D",
            ),
            up_block_types=("DreamLiteCrossAttnUpBlock2D", "DreamLiteUpBlock2D"),
            block_out_channels=(32, 64),
            cross_attention_dim=CROSS_ATTN_DIM,
            attention_head_dim=8,
            layers_per_block=1,
            norm_num_groups=8,
            transformer_layers_per_block=1,
        )

        torch.manual_seed(0)
        vae = AutoencoderTiny(
            in_channels=3,
            out_channels=3,
            encoder_block_out_channels=(32, 32),
            decoder_block_out_channels=(32, 32),
            num_encoder_blocks=(1, 1),
            num_decoder_blocks=(1, 1),
            latent_channels=4,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        torch.manual_seed(0)
        text_encoder = build_tiny_text_encoder()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
        processor = Qwen3VLProcessor.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
            "vae": vae,
            "unet": unet,
            "scheduler": scheduler,
        }
