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

"""Original configuration helpers and model presets for the kandinsky assembly recipe."""

PRIOR_CONFIG = {}

UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_image",
    "addition_embed_type_num_heads": 64,
    "attention_head_dim": 64,
    "block_out_channels": [384, 768, 1152, 1536],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": False,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 768,
    "cross_attention_norm": None,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": 1024,
    "encoder_hid_dim_type": "text_image_proj",
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 3,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 8,
    "projection_class_embeddings_input_dim": None,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "scale_shift",
    "sample_size": 64,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "up_block_types": [
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "upcast_attention": False,
    "use_linear_projection": False,
}

INPAINT_UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_image",
    "addition_embed_type_num_heads": 64,
    "attention_head_dim": 64,
    "block_out_channels": [384, 768, 1152, 1536],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": None,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 768,
    "cross_attention_norm": None,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": 1024,
    "encoder_hid_dim_type": "text_image_proj",
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 9,
    "layers_per_block": 3,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 8,
    "projection_class_embeddings_input_dim": None,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "scale_shift",
    "sample_size": 64,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "up_block_types": [
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "upcast_attention": False,
    "use_linear_projection": False,
}

MOVQ_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "AttnDownEncoderBlock2D"),
    "up_block_types": ("AttnUpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    "num_vq_embeddings": 16384,
    "block_out_channels": (128, 256, 256, 512),
    "vq_embed_dim": 4,
    "layers_per_block": 2,
    "norm_type": "spatial",
}

__all__ = ["INPAINT_UNET_CONFIG", "MOVQ_CONFIG", "PRIOR_CONFIG", "UNET_CONFIG"]
