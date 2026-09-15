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

"""Original configuration helpers and model presets for the consistency assembly recipe."""

TEST_UNET_CONFIG = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "num_class_embeds": 1000,
    "block_out_channels": [32, 64],
    "attention_head_dim": 8,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "attn_norm_num_groups": 32,
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

IMAGENET_64_UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 3,
    "num_class_embeds": 1000,
    "block_out_channels": [192, 192 * 2, 192 * 3, 192 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "attn_norm_num_groups": 32,
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

LSUN_256_UNET_CONFIG = {
    "sample_size": 256,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "num_class_embeds": None,
    "block_out_channels": [256, 256, 256 * 2, 256 * 2, 256 * 4, 256 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "default",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

CD_SCHEDULER_CONFIG = {
    "num_train_timesteps": 40,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
}

CT_IMAGENET_64_SCHEDULER_CONFIG = {
    "num_train_timesteps": 201,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
}

CT_LSUN_256_SCHEDULER_CONFIG = {
    "num_train_timesteps": 151,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
}

__all__ = [
    "CD_SCHEDULER_CONFIG",
    "CT_IMAGENET_64_SCHEDULER_CONFIG",
    "CT_LSUN_256_SCHEDULER_CONFIG",
    "IMAGENET_64_UNET_CONFIG",
    "LSUN_256_UNET_CONFIG",
    "TEST_UNET_CONFIG",
]
