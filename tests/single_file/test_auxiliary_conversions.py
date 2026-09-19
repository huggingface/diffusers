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

import importlib
import json

import pytest
import torch

import diffusers
from diffusers.loaders.conversion import get_conversion


CASES = [
    (
        "AudioLDM2ProjectionModel",
        {
            "text_encoder_dim": 8,
            "text_encoder_1_dim": 16,
            "langauge_model_dim": 32,
            "use_learned_position_embedding": True,
            "max_seq_length": 16,
        },
    ),
    (
        "AudioLDM2UNet2DConditionModel",
        {
            "block_out_channels": (32, 64),
            "layers_per_block": 1,
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": ((8, 16), (8, 16)),
        },
    ),
    (
        "AutoencoderKLHunyuanImage",
        {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "block_out_channels": (32, 64),
            "layers_per_block": 1,
            "spatial_compression_ratio": 2,
            "sample_size": 16,
        },
    ),
    (
        "AutoencoderKLHunyuanImageRefiner",
        {
            "latent_channels": 4,
            "block_out_channels": (32, 64),
            "layers_per_block": 1,
            "spatial_compression_ratio": 2,
            "temporal_compression_ratio": 2,
        },
    ),
    (
        "HunyuanDiT2DControlNetModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "in_channels": 4,
            "patch_size": 2,
            "sample_size": 8,
            "hidden_size": 32,
            "transformer_num_layers": 4,
            "cross_attention_dim": 16,
            "cross_attention_dim_t5": 32,
            "pooled_projection_dim": 16,
            "text_len": 4,
            "text_len_t5": 8,
        },
    ),
    (
        "Kandinsky3UNet",
        {
            "block_out_channels": (32, 64, 64, 64),
            "time_embedding_dim": 128,
            "layers_per_block": 1,
            "attention_head_dim": 32,
            "cross_attention_dim": 32,
            "encoder_hid_dim": 32,
            "groups": 8,
        },
    ),
    (
        "SD3ControlNetModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 2,
            "in_channels": 4,
            "out_channels": 4,
            "joint_attention_dim": 16,
            "caption_projection_dim": 16,
            "pooled_projection_dim": 16,
            "pos_embed_max_size": 8,
            "qk_norm": "rms_norm",
            "dual_attention_layers": (0,),
        },
    ),
    (
        "SanaControlNetModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "num_layers": 2,
            "num_cross_attention_heads": 2,
            "cross_attention_head_dim": 16,
            "cross_attention_dim": 32,
            "caption_channels": 32,
            "in_channels": 4,
            "out_channels": 4,
        },
    ),
    (
        "UNetFlatConditionModel",
        {
            "block_out_channels": (32, 64),
            "layers_per_block": 1,
            "down_block_types": ("CrossAttnDownBlockFlat", "DownBlockFlat"),
            "up_block_types": ("UpBlockFlat", "CrossAttnUpBlockFlat"),
            "cross_attention_dim": 16,
        },
    ),
    (
        "UnCLIPTextProjModel",
        {"clip_extra_context_tokens": 2, "clip_embeddings_dim": 8, "time_embed_dim": 16, "cross_attention_dim": 16},
    ),
    (
        "SpectrogramContEncoder",
        {
            "input_dims": 8,
            "targets_context_length": 4,
            "d_model": 16,
            "dropout_rate": 0,
            "num_layers": 1,
            "num_heads": 2,
            "d_kv": 8,
            "d_ff": 32,
            "feed_forward_proj": "gated-gelu",
        },
    ),
    (
        "SpectrogramNotesEncoder",
        {
            "max_length": 8,
            "vocab_size": 16,
            "d_model": 16,
            "dropout_rate": 0,
            "num_layers": 1,
            "num_heads": 2,
            "d_kv": 8,
            "d_ff": 32,
            "feed_forward_proj": "gated-gelu",
        },
    ),
    (
        "AutoencoderRAE",
        {
            "encoder_type": "siglip2",
            "encoder_hidden_size": 64,
            "encoder_num_hidden_layers": 1,
            "encoder_patch_size": 16,
            "encoder_input_size": 32,
            "decoder_hidden_size": 32,
            "decoder_num_hidden_layers": 1,
            "decoder_num_attention_heads": 2,
            "decoder_intermediate_size": 64,
        },
    ),
    ("UNet1DModel", {}),
    (
        "ChronoEditTransformer3DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 1,
        },
    ),
    (
        "QwenImageTransformer2DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "joint_attention_dim": 16,
            "num_layers": 1,
            "axes_dims_rope": [2, 2, 4],
            "use_additional_t_cond": True,
        },
    ),
    (
        "MotifVideoTransformer3DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "text_embed_dim": 16,
            "num_layers": 1,
            "num_single_layers": 2,
            "num_decoder_layers": 1,
            "rope_axes_dim": [2, 2, 4],
        },
    ),
    (
        "MotifVideoTransformer3DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "text_embed_dim": 16,
            "image_embed_dim": 8,
            "num_layers": 1,
            "num_single_layers": 2,
            "num_decoder_layers": 1,
            "rope_axes_dim": [2, 2, 4],
            "qk_norm": "layer_norm",
            "enable_text_cross_attention_dual": True,
            "enable_text_cross_attention_single": True,
        },
    ),
    ("LTXLatentUpsamplerModel", {"in_channels": 4, "mid_channels": 32, "num_blocks_per_stage": 1}),
    (
        "SkyReelsV2Transformer3DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 1,
            "inject_sample_info": True,
        },
    ),
    (
        "SkyReelsV2Transformer3DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 1,
            "image_dim": 16,
            "added_kv_proj_dim": 16,
            "pos_embed_seq_len": 4,
        },
    ),
    (
        "HunyuanImageTransformer2DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 1,
            "num_single_layers": 1,
            "num_refiner_layers": 1,
            "in_channels": 4,
            "out_channels": 4,
            "text_embed_dim": 16,
            "rope_axes_dim": [4, 4],
        },
    ),
    (
        "HunyuanImageTransformer2DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 1,
            "num_single_layers": 1,
            "num_refiner_layers": 1,
            "in_channels": 4,
            "out_channels": 4,
            "text_embed_dim": 16,
            "text_embed_2_dim": 8,
            "rope_axes_dim": [4, 4],
            "use_meanflow": True,
            "guidance_embeds": True,
        },
    ),
    (
        "SparseControlNetModel",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["CrossAttnDownBlockMotion", "DownBlockMotion"],
            "layers_per_block": 1,
            "cross_attention_dim": 16,
            "motion_num_attention_heads": 2,
        },
    ),
    (
        "SparseControlNetModel",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["CrossAttnDownBlockMotion", "DownBlockMotion"],
            "layers_per_block": 1,
            "cross_attention_dim": 16,
            "motion_num_attention_heads": 2,
            "use_simplified_condition_embedding": False,
            "conditioning_embedding_out_channels": [8, 16],
            "transformer_layers_per_mid_block": 2,
        },
    ),
    (
        "CogView4Transformer2DModel",
        {
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "text_embed_dim": 16,
            "condition_dim": 8,
            "original_format": "megatron",
        },
    ),
    (
        "ConsistencyDecoderVAE",
        {
            "encoder_block_out_channels": [32, 64],
            "encoder_down_block_types": ["DownEncoderBlock2D"] * 2,
            "encoder_layers_per_block": 1,
            "decoder_block_out_channels": [32, 64],
            "decoder_down_block_types": ["ResnetDownsampleBlock2D"] * 2,
            "decoder_up_block_types": ["ResnetUpsampleBlock2D"] * 2,
            "decoder_layers_per_block": 1,
        },
    ),
    (
        "UNet2DConditionModel",
        {
            "block_out_channels": [32, 64, 64],
            "down_block_types": ["KDownBlock2D", "KCrossAttnDownBlock2D", "KCrossAttnDownBlock2D"],
            "up_block_types": ["KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"],
            "mid_block_type": None,
            "layers_per_block": 2,
            "cross_attention_dim": 16,
            "attention_head_dim": 8,
            "time_embedding_type": "fourier",
            "norm_num_groups": None,
            "time_cond_proj_dim": 16,
            "conv_in_kernel": 1,
            "conv_out_kernel": 1,
        },
    ),
    (
        "UNet2DConditionModel",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["SimpleCrossAttnDownBlock2D", "ResnetDownsampleBlock2D"],
            "up_block_types": ["ResnetUpsampleBlock2D", "SimpleCrossAttnUpBlock2D"],
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "layers_per_block": 1,
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "encoder_hid_dim": 16,
            "addition_embed_type": "text",
            "addition_embed_type_num_heads": 2,
            "cross_attention_norm": "group_norm",
        },
    ),
    (
        "UNet2DConditionModel",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["SimpleCrossAttnDownBlock2D", "ResnetDownsampleBlock2D"],
            "up_block_types": ["ResnetUpsampleBlock2D", "SimpleCrossAttnUpBlock2D"],
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "layers_per_block": 1,
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "encoder_hid_dim": 16,
            "addition_embed_type": "text",
            "addition_embed_type_num_heads": 2,
            "cross_attention_norm": "group_norm",
            "only_cross_attention": True,
            "class_embed_type": "timestep",
        },
    ),
    (
        "JoyImageEditTransformer3DModel",
        {"hidden_size": 32, "num_attention_heads": 4, "text_dim": 16, "num_layers": 1, "rope_dim_list": [2, 2, 4]},
    ),
    (
        "JoyImageEditPlusTransformer3DModel",
        {"hidden_size": 32, "num_attention_heads": 4, "text_dim": 16, "num_layers": 1, "rope_dim_list": [2, 2, 4]},
    ),
    (
        "Transformer2DModel",
        {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 1,
            "num_vector_embeds": 16,
            "sample_size": 4,
            "cross_attention_dim": 16,
            "num_embeds_ada_norm": 10,
            "norm_type": "ada_norm",
            "attention_bias": True,
            "activation_fn": "gelu",
        },
    ),
    (
        "VQModel",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
            "up_block_types": ["AttnUpDecoderBlock2D", "UpDecoderBlock2D"],
            "layers_per_block": 1,
            "latent_channels": 4,
            "num_vq_embeddings": 16,
        },
    ),
    (
        "ShapERenderer",
        {
            "d_latent": 16,
            "d_hidden": 16,
            "param_names": ["nerstf.mlp.0.weight"],
            "param_shapes": [[16, 93]],
            "n_hidden_layers": 2,
            "insert_direction_at": 1,
        },
    ),
    (
        "LTX2DurationHead",
        {
            "video_cross_attention_dim": 16,
            "audio_cross_attention_dim": 8,
            "pooler_hidden_dim": 16,
            "mlp_hidden_dim": 16,
        },
    ),
    ("LTX2LatentUpsamplerModel", {"in_channels": 4, "mid_channels": 32, "num_blocks_per_stage": 1}),
    (
        "LTX2LatentUpsamplerModel",
        {
            "in_channels": 4,
            "mid_channels": 32,
            "num_blocks_per_stage": 1,
            "spatial_upsample": False,
            "temporal_upsample": True,
        },
    ),
    (
        "UNet2DConditionModel",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
            "layers_per_block": 1,
            "cross_attention_dim": 32,
            "attention_type": "gated",
        },
    ),
    (
        "UNet2DConditionModel",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
            "layers_per_block": 1,
            "cross_attention_dim": 32,
            "attention_type": "gated-text-image",
        },
    ),
    (
        "VQModel",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["DownEncoderBlock2D"] * 2,
            "up_block_types": ["UpDecoderBlock2D"] * 2,
            "layers_per_block": 1,
            "norm_type": "spatial",
            "latent_channels": 4,
            "num_vq_embeddings": 16,
        },
    ),
    (
        "UniDiffuserModel",
        {
            "text_dim": 8,
            "clip_img_dim": 8,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 2,
            "sample_size": 8,
            "patch_size": 2,
            "activation_fn": "gelu",
            "use_data_type_embedding": True,
        },
    ),
    (
        "UniDiffuserTextDecoder",
        {
            "prefix_length": 4,
            "prefix_inner_dim": 8,
            "prefix_hidden_dim": 8,
            "vocab_size": 32,
            "n_positions": 16,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
        },
    ),
    (
        "LTX2TextConnectors",
        {
            "caption_channels": 16,
            "text_proj_in_factor": 2,
            "video_connector_num_attention_heads": 2,
            "video_connector_attention_head_dim": 8,
            "video_connector_num_layers": 1,
            "audio_connector_num_attention_heads": 2,
            "audio_connector_attention_head_dim": 8,
            "audio_connector_num_layers": 1,
        },
    ),
    (
        "LTX2TextConnectors",
        {
            "caption_channels": 16,
            "text_proj_in_factor": 2,
            "video_connector_num_attention_heads": 2,
            "video_connector_attention_head_dim": 8,
            "video_connector_num_layers": 1,
            "audio_connector_num_attention_heads": 2,
            "audio_connector_attention_head_dim": 8,
            "audio_connector_num_layers": 1,
            "per_modality_projections": True,
            "video_hidden_dim": 16,
            "audio_hidden_dim": 16,
            "video_gated_attn": True,
            "audio_gated_attn": True,
            "proj_bias": True,
        },
    ),
    (
        "LTX2Vocoder",
        {
            "in_channels": 8,
            "hidden_channels": 32,
            "upsample_kernel_sizes": [4, 4],
            "upsample_factors": [2, 2],
            "resnet_kernel_sizes": [3],
            "resnet_dilations": [[1, 3]],
        },
    ),
    (
        "LTX2VocoderWithBWE",
        {
            "in_channels": 8,
            "hidden_channels": 32,
            "upsample_kernel_sizes": [4, 4],
            "upsample_factors": [2, 2],
            "resnet_kernel_sizes": [3],
            "resnet_dilations": [[1, 3]],
            "bwe_in_channels": 8,
            "bwe_hidden_channels": 32,
            "bwe_upsample_kernel_sizes": [4],
            "bwe_upsample_factors": [2],
            "bwe_resnet_kernel_sizes": [3],
            "bwe_resnet_dilations": [[1, 3]],
            "filter_length": 16,
            "window_length": 16,
            "num_mel_channels": 8,
        },
    ),
    (
        "LTX2VideoDiffusionDecoderModel",
        {
            "latent_channels": 8,
            "decoder_head_dim": 8,
            "decoder_stage_channels": [32, 16, 8, 8, 8],
            "decoder_stage_depths": [1, 1, 1, 1, 1],
            "decoder_upsample_channel_reductions": [2, 2, 1, 1],
            "decoder_t_emb_dim": 16,
        },
    ),
    (
        "LTX2VideoDiffusionDecoderModel",
        {
            "latent_channels": 8,
            "decoder_head_dim": 8,
            "decoder_stage_channels": [32, 16, 8, 8, 8],
            "decoder_stage_depths": [1, 1, 1, 1, 1],
            "decoder_upsample_channel_reductions": [2, 2, 1, 1],
            "decoder_t_emb_dim": 16,
            "original_format": "ltx2_diffusion_decoder_gated",
        },
    ),
    ("WuerstchenPrior", {"c_in": 4, "c": 16, "c_cond": 8, "c_r": 8, "depth": 2, "nhead": 2}),
    (
        "WuerstchenDiffNeXt",
        {
            "c_hidden": [16, 32],
            "nhead": [2, 2],
            "blocks": [1, 2],
            "level_config": ["CT", "CTA"],
            "inject_effnet": [False, True],
            "c_cond": 16,
            "clip_embd": 8,
            "effnet_embd": 4,
        },
    ),
    ("PaellaVQModel", {"levels": 2, "bottleneck_blocks": 2, "embed_dim": 16, "num_vq_embeddings": 16}),
    ("T2IAdapter", {"channels": [16, 32], "num_res_blocks": 2, "adapter_type": "full_adapter"}),
    ("T2IAdapter", {"channels": [16, 32], "num_res_blocks": 2, "adapter_type": "light_adapter"}),
    ("T2IAdapter", {"channels": [16, 32, 32, 32], "num_res_blocks": 2, "adapter_type": "full_adapter_xl"}),
    (
        "AutoencoderKLFlux2",
        {
            "block_out_channels": [32, 64],
            "decoder_block_out_channels": [32, 32],
            "layers_per_block": 1,
            "down_block_types": ["DownEncoderBlock2D"] * 2,
            "up_block_types": ["UpDecoderBlock2D"] * 2,
        },
    ),
    (
        "I2VGenXLUNet",
        {
            "block_out_channels": [32, 64],
            "down_block_types": ["CrossAttnDownBlock3D", "DownBlock3D"],
            "up_block_types": ["UpBlock3D", "CrossAttnUpBlock3D"],
            "layers_per_block": 1,
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
        },
    ),
    (
        "UNet2DModel",
        {
            "block_out_channels": (32, 64),
            "down_block_types": ("DownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "UpBlock2D"),
            "layers_per_block": 1,
        },
    ),
    (
        "UNet2DModel",
        {
            "block_out_channels": (32, 64),
            "down_block_types": ("DownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "UpBlock2D"),
            "layers_per_block": 1,
            "original_format": "ldm",
        },
    ),
    (
        "UNet2DModel",
        {
            "block_out_channels": (32, 64),
            "down_block_types": ("ResnetDownsampleBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "layers_per_block": 1,
            "downsample_type": "resnet",
            "upsample_type": "resnet",
            "original_format": "consistency",
        },
    ),
    (
        "UNet2DModel",
        {
            "block_out_channels": (32, 64),
            "down_block_types": ("SkipDownBlock2D", "AttnSkipDownBlock2D"),
            "up_block_types": ("AttnSkipUpBlock2D", "SkipUpBlock2D"),
            "layers_per_block": 1,
            "time_embedding_type": "fourier",
        },
    ),
    (
        "AutoencoderRAE",
        {
            "encoder_type": "dinov2",
            "encoder_hidden_size": 64,
            "encoder_num_hidden_layers": 1,
            "encoder_patch_size": 16,
            "encoder_input_size": 32,
            "decoder_hidden_size": 32,
            "decoder_num_hidden_layers": 1,
            "decoder_num_attention_heads": 2,
            "decoder_intermediate_size": 64,
        },
    ),
    (
        "AutoencoderRAE",
        {
            "encoder_type": "mae",
            "encoder_hidden_size": 64,
            "encoder_num_hidden_layers": 1,
            "encoder_patch_size": 16,
            "encoder_input_size": 32,
            "decoder_hidden_size": 32,
            "decoder_num_hidden_layers": 1,
            "decoder_num_attention_heads": 2,
            "decoder_intermediate_size": 64,
        },
    ),
    (
        "AutoencoderKLMochi",
        {
            "encoder_block_out_channels": (32, 64),
            "decoder_block_out_channels": (32, 64),
            "layers_per_block": (1, 1, 1),
            "temporal_expansions": (2,),
            "spatial_expansions": (2,),
            "add_attention_block": (False, True, True),
        },
    ),
    (
        "AutoencoderKLHunyuanVideo",
        {
            "block_out_channels": (32, 64),
            "down_block_types": ("HunyuanVideoDownBlock3D",) * 2,
            "up_block_types": ("HunyuanVideoUpBlock3D",) * 2,
            "layers_per_block": 1,
            "spatial_compression_ratio": 2,
        },
    ),
    (
        "AutoencoderKLHunyuanVideo15",
        {
            "block_out_channels": (8, 16, 16),
            "layers_per_block": 1,
            "latent_channels": 4,
            "spatial_compression_ratio": 4,
            "temporal_compression_ratio": 2,
        },
    ),
    (
        "AutoencoderKLCosmos",
        {
            "encoder_block_out_channels": (8, 16, 16),
            "decode_block_out_channels": (8, 16, 16),
            "num_layers": 1,
            "resolution": 16,
            "patch_size": 2,
            "attention_resolutions": (8, 4),
            "spatial_compression_ratio": 4,
            "temporal_compression_ratio": 4,
        },
    ),
    (
        "HunyuanVideo15Transformer3DModel",
        {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 2,
            "num_refiner_layers": 1,
            "text_embed_dim": 16,
            "text_embed_2_dim": 16,
            "image_embed_dim": 16,
            "rope_axes_dim": (2, 2, 4),
        },
    ),
    (
        "AnimaTextConditioner",
        {
            "source_dim": 8,
            "target_dim": 8,
            "model_dim": 16,
            "num_layers": 2,
            "num_attention_heads": 2,
            "target_vocab_size": 32,
        },
    ),
    ("AutoencoderKLQwenImage", {"base_dim": 4, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 1}),
    (
        "ControlNetModel",
        {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "layers_per_block": 1,
            "cross_attention_dim": 32,
        },
    ),
    ("MotionAdapter", {"block_out_channels": (32, 64), "motion_layers_per_block": 1}),
    (
        "StableCascadeUNet",
        {
            "block_out_channels": (16, 32),
            "conditioning_dim": 16,
            "num_attention_heads": (2, 4),
            "down_num_layers_per_block": (1, 2),
            "up_num_layers_per_block": (2, 1),
        },
    ),
    (
        "ZImageControlNetModel",
        {
            "control_layers_places": [0, 2],
            "control_in_dim": 4,
            "dim": 16,
            "n_refiner_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 2,
        },
    ),
    (
        "StableAudioDiTModel",
        {
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_attention_heads": 1,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "time_proj_dim": 16,
            "global_states_input_dim": 16,
            "cross_attention_input_dim": 16,
        },
    ),
    ("StableAudioProjectionModel", {"text_encoder_dim": 16, "conditioning_dim": 16, "min_value": 0, "max_value": 60}),
    ("StableAudio3DurationEmbedder", {"output_dim": 8, "fourier_dim": 16}),
    (
        "AutoencoderSAME",
        {
            "audio_channels": 2,
            "patch_size": 4,
            "encoder_channels": 8,
            "encoder_c_mults": (2, 2),
            "encoder_strides": (2, 2),
            "encoder_transformer_depths": (1, 2),
            "latent_dim": 4,
            "dim_heads": 4,
        },
    ),
    (
        "LongCatAudioDiTTransformer",
        {"dit_dim": 16, "dit_depth": 2, "dit_heads": 2, "dit_text_dim": 16, "latent_dim": 4},
    ),
    (
        "LongCatAudioDiTVae",
        {"channels": 8, "c_mults": [1, 2], "strides": [2, 2], "latent_dim": 4, "encoder_latent_dim": 8},
    ),
    ("MiniMaxMusic3ConditionEncoder", {"condition_hidden_dim": 8, "num_condition_layers": 2, "out_dim": 16}),
    (
        "MiniMaxMusic3Vocoder",
        {"latent_channels": 8, "decoder_input_dim": 16, "decoder_hidden_dim": 32, "upsampling_ratios": (2, 2)},
    ),
    (
        "MiniMaxMusic3RVQDepthDecoder",
        {
            "hidden_size": 16,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "audio_vocab_size": 16,
            "num_codebooks": 3,
        },
    ),
    (
        "AceStepTransformer1DModel",
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "in_channels": 12,
            "audio_acoustic_hidden_dim": 4,
        },
    ),
    (
        "AceStepConditionEncoder",
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "text_hidden_dim": 8,
            "timbre_hidden_dim": 4,
            "num_lyric_encoder_hidden_layers": 1,
            "num_timbre_encoder_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
    ),
    (
        "AceStepAudioTokenizer",
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "audio_acoustic_hidden_dim": 4,
            "fsq_dim": 16,
            "num_attention_pooler_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
    ),
    (
        "AceStepAudioTokenDetokenizer",
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "audio_acoustic_hidden_dim": 4,
            "num_attention_pooler_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
    ),
    (
        "UVit2DModel",
        {
            "hidden_size": 16,
            "cond_embed_dim": 8,
            "micro_cond_embed_dim": 16,
            "encoder_hidden_size": 16,
            "vocab_size": 32,
            "codebook_size": 32,
            "in_channels": 16,
            "block_out_channels": 16,
            "num_res_blocks": 1,
            "block_num_heads": 2,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
        },
    ),
    (
        "VQModel",
        {
            "block_out_channels": (16, 32),
            "down_block_types": ("DownEncoderBlock2D",) * 2,
            "up_block_types": ("UpDecoderBlock2D",) * 2,
            "norm_num_groups": 8,
            "latent_channels": 4,
            "num_vq_embeddings": 16,
        },
    ),
    (
        "AutoencoderTiny",
        {
            "encoder_block_out_channels": (8, 8),
            "decoder_block_out_channels": (8, 8),
            "num_encoder_blocks": (1, 2),
            "num_decoder_blocks": (2, 1),
        },
    ),
    (
        "T5FilmDecoder",
        {"input_dims": 4, "targets_length": 16, "d_model": 16, "num_layers": 2, "num_heads": 2, "d_kv": 8, "d_ff": 32},
    ),
]


CASES.extend(
    (name, {**config, "original_format": "consistency_decoder_jit"})
    for name, config in list(CASES)
    if name == "ConsistencyDecoderVAE"
)


@pytest.mark.parametrize("model_class,config", CASES, ids=[name for name, _ in CASES])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["float32", "float16", "bfloat16"]
)
def test_auxiliary_model_conversion(model_class, config, dtype, tmp_path):
    model_config = {
        key: value for key, value in config.items() if key not in ("original_format", "transformers_version")
    }
    internal_modules = {
        "UNetFlatConditionModel": "diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet",
        "UnCLIPTextProjModel": "diffusers.pipelines.deprecated.unclip.text_proj",
        "SpectrogramContEncoder": "diffusers.pipelines.deprecated.spectrogram_diffusion.continuous_encoder",
        "SpectrogramNotesEncoder": "diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder",
        "LTXLatentUpsamplerModel": "diffusers.pipelines.ltx.modeling_latent_upsampler",
        "ShapERenderer": "diffusers.pipelines.shap_e.renderer",
        "UniDiffuserModel": "diffusers.pipelines.deprecated.unidiffuser.modeling_uvit",
        "UniDiffuserTextDecoder": "diffusers.pipelines.deprecated.unidiffuser.modeling_text_decoder",
        "LTX2TextConnectors": "diffusers.pipelines.ltx2.connectors",
        "LTX2Vocoder": "diffusers.pipelines.ltx2.vocoder",
        "LTX2VocoderWithBWE": "diffusers.pipelines.ltx2.vocoder",
        "LTX2DurationHead": "diffusers.pipelines.ltx2.duration_head",
        "LTX2LatentUpsamplerModel": "diffusers.pipelines.ltx2.latent_upsampler",
        "WuerstchenPrior": "diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_prior",
        "WuerstchenDiffNeXt": "diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_diffnext",
        "PaellaVQModel": "diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model",
    }
    module = importlib.import_module(internal_modules[model_class]) if model_class in internal_modules else diffusers
    model = getattr(module, model_class)(**model_config).to(dtype=dtype)
    state = model.state_dict()
    if model_class == "ShapERenderer":
        from diffusers.loaders.conversion.shap_e_tables import create_mc_lookup_table

        state["mesh_decoder.cases"], state["mesh_decoder.masks"] = create_mc_lookup_table()
    conversion_config = dict(model.config)
    conversion_config.update(
        {key: value for key, value in config.items() if key in ("original_format", "transformers_version")}
    )
    conversion = get_conversion(model_class, conversion_config)
    assert conversion.diffusers_keys == set(state)
    if model_class == "UniDiffuserModel":
        for p in ("weight", "bias"):
            state[f"transformer.pos_embed.proj.{p}"] = state[f"vae_img_in.proj.{p}"].clone()
    original = conversion.to_original(state)
    restored = conversion.to_diffusers(original)
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0)
    if model_class == "AutoencoderRAE" and config["encoder_type"] == "siglip2" and dtype == torch.float32:
        model.eval()
        sample = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            expected = model.encode(sample).latent
            model.load_state_dict(restored)
            actual = model.encode(sample).latent
        assert actual.shape == (1, 64, 2, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if dtype == torch.float32 and hasattr(type(model), "from_single_file"):
        (tmp_path / "config.json").write_text(json.dumps(conversion_config), encoding="utf-8")
        loaded = type(model).from_single_file(original, config=str(tmp_path), local_files_only=True)
        for key in state:
            torch.testing.assert_close(loaded.state_dict()[key], state[key], rtol=0, atol=0)
