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

"""Original configuration helpers and model presets for the unclip assembly recipe."""

PRIOR_CONFIG = {}

DECODER_CONFIG = {
    "sample_size": 64,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ),
    "up_block_types": (
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 3,
    "out_channels": 6,
    "cross_attention_dim": 1536,
    "class_embed_type": "identity",
    "attention_head_dim": 64,
    "resnet_time_scale_shift": "scale_shift",
}

SUPER_RES_UNET_FIRST_STEPS_CONFIG = {
    "sample_size": 256,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
    ),
    "up_block_types": (
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 6,
    "out_channels": 3,
    "add_attention": False,
}

SUPER_RES_UNET_LAST_STEP_CONFIG = {
    "sample_size": 256,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
    ),
    "up_block_types": (
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 6,
    "out_channels": 3,
    "add_attention": False,
}

__all__ = ["DECODER_CONFIG", "PRIOR_CONFIG", "SUPER_RES_UNET_FIRST_STEPS_CONFIG", "SUPER_RES_UNET_LAST_STEP_CONFIG"]
